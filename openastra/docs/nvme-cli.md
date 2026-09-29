# Driving OpenASTRA with nvme-cli

Host tutorial derived from the OpenASTRA firmware ABI
(`AISSDFTL/src/nvme/*`, `AISSDFTL/src/gemm/*`, `memory_map.h`).

**Hierarchy:** Host → NVMe MoE command (`0x80`) → TEP → native GEMM tiles → FPGA NPU.

The TEP execution path was functionally validated on the OpenASTRA FPGA
prototype for the supported multi-tile configuration.

> **Environment note:** `nvme-cli` is not installed on the Windows documentation
> host. Option names match upstream `nvme-cli io-passthru`. Confirm with
> `nvme io-passthru --help` on the Linux PCIe host. **Commands below were not
> executed against hardware from this documentation environment.**

---

## Prerequisites

- OpenASTRA / Cosmos+ board on a Linux PCIe host
- Bitstream: `OpenSSD2_hw_platform_0/OpenSSD2.bit`
- TEP-enabled `AISSDFTL.elf` on `ps7_cortexa9_0`
- `nvme-cli`
- Expert weights pre-written to flash at a slice-aligned LBA

FPGA / FW load (manual):

1. Program `OpenSSD2.bit` (Xilinx SDK / Vivado Hardware Manager 2019.1).
2. Load TEP-enabled `AISSDFTL.elf`.
3. Confirm enumeration with `nvme list`.

---

## Check NVMe Enumeration

```bash
nvme list

CTRL=/dev/nvme0
DEV=/dev/nvme0n1
NSID=1

nvme id-ctrl "$CTRL"
nvme id-ns "$DEV"
```

Identify strings (firmware): MN=`Cosmos+ OpenSSD`, SN=`SSDD515T`, FR=`TYPE0005`, NN=`1`.

---

## OpenASTRA NVMe Commands

| Opcode | Name | Direction | Payload |
|--------|------|-----------|---------|
| `0x01` | Write | host → NAND | standard |
| `0x02` | Read | NAND → host | standard |
| `0x80` | MoE / TEP execute | none (on-device) | none |
| `0x81` | ACT_WRITE | host → **IFMAP DRAM** | `--write --data-len` |
| `0x82` | ACT_READ | **OFMAP DRAM** → host | `--read --data-len` |

Weights remain NAND-backed. Activations / results use controller DRAM.

### Can stock nvme-cli drive the full flow?

**YES** (for the supported sizes below), after programming bitstream + TEP ELF
and placing expert weights.

---

## MoE Command Encoding (`0x80`)

Linux passthrough-accessible DWORDs only (`cdw2/3/10–15`).

| DW | Field | Meaning | Notes |
|----|-------|---------|-------|
| CDW3 | `NUMR` | rows `M` | supported `16`; `0` → default `16` |
| CDW10 | expert LBA | weight base | must be `% 4 == 0` |
| CDW11 | LBA upper | — | `0` |
| CDW12 | IFMAP offset | from `0x01000000` | halfword-aligned |
| CDW13 | OFMAP offset | from `0x02000000` | halfword-aligned |
| CDW14 | `DLEN` | columns `N` | `N=32*T`, `1≤T≤64`; `0` → default `32` |

`NUMR` / `DLEN` are full 32-bit values (no bit packing):

```text
cdw3  = NUMR     # 16 -> 0x10
cdw14 = DLEN     # 64 -> 0x40
```

---

## Activation Command Encoding (`0x81` / `0x82`)

| DW | Field | Meaning |
|----|-------|---------|
| CDW10 | `ActOffset` | byte offset from region base |
| CDW12 | `ActLen` | transfer length in bytes (`1 .. 4096`) |

| Opcode | Region base | Allowed range |
|--------|-------------|---------------|
| `0x81` | `IFMAP_BUFFER_BASE_ADDR` (`0x01000000`) | entirely inside IFMAP (`…` to `0x02000000`) |
| `0x82` | `OFMAP_BUFFER_BASE_ADDR` (`0x02000000`) | entirely inside OFMAP (`…` to `0x03000000`) |

DMA primitive: existing `set_direct_rx_dma` / `set_direct_tx_dma` +
`check_direct_*_dma_done` (same family as Identify).

---

## Weight Storage Contract

```text
baseSlice = EXPERT_LBA / 4          # EXPERT_LBA % 4 == 0
tile i    = baseSlice + i           # FLASH_PAGES_PER_TILE = 1
```

Firmware does not transpose weights. Prepare tiles offline; write with
standard NVMe write.

---

## Supported TEP Configuration

| Item | Value |
|------|--------|
| `M` | 16 |
| `K` | 32 |
| `N` | `32 * T`, `1 ≤ T ≤ 64` |
| Weight slots | 2 |
| IFMAP tile | `0x100` bytes |
| OFMAP / tile | `0x400` bytes |

---

## Two-Tile TEP Example (`M=16`, `K=32`, `N=64`)

```bash
DEV=/dev/nvme0n1
NSID=1
EXPERT_LBA=<SLICE_ALIGNED_LBA>   # deployment-specific; multiple of 4
```

### 1. Write IFMAP (256 bytes)

```bash
sudo nvme io-passthru "$DEV" \
  --opcode=0x81 \
  --namespace-id="$NSID" \
  --cdw10=0x0 \
  --cdw12=0x100 \
  --write \
  --data-len=256 \
  --input-file=input.bin
```

| Field | Meaning |
|-------|---------|
| `--opcode=0x81` | ACT_WRITE → IFMAP |
| `--cdw10=0` | IFMAP offset 0 |
| `--cdw12=0x100` | 256 bytes |
| `--write --data-len=256` | host payload |

### 2. Execute two-tile TEP

```bash
sudo nvme io-passthru "$DEV" \
  --opcode=0x80 \
  --namespace-id="$NSID" \
  --cdw3=0x10 \
  --cdw10="$EXPERT_LBA" \
  --cdw11=0x0 \
  --cdw12=0x0 \
  --cdw13=0x0 \
  --cdw14=0x40
```

| Field | Meaning |
|-------|---------|
| `--opcode=0x80` | MoE / TEP execute |
| `--cdw3=0x10` | `NUMR=16` |
| `--cdw10` | expert base LBA |
| `--cdw12=0` | IFMAP offset 0 |
| `--cdw13=0` | OFMAP offset 0 |
| `--cdw14=0x40` | `DLEN=64` → `tileCount=2` |

On the FPGA: decode → two tiles → flash weight reads → out-of-order
flash-ready NPU GEMM → contiguous OFMAP (`2 × 0x400`) → one NVMe completion.

### 3. Read OFMAP (2048 bytes)

```bash
sudo nvme io-passthru "$DEV" \
  --opcode=0x82 \
  --namespace-id="$NSID" \
  --cdw10=0x0 \
  --cdw12=0x800 \
  --read \
  --data-len=2048 \
  --raw-binary > output.bin
```

| Field | Meaning |
|-------|---------|
| `--opcode=0x82` | ACT_READ ← OFMAP |
| `--cdw10=0` | OFMAP offset 0 |
| `--cdw12=0x800` | 2048 bytes (`16×64×2`) |

Expected: `output.bin` size **2048**.

---

## Remaining Placeholders

| Placeholder | Why |
|-------------|-----|
| `$DEV` / `$NSID` | Host enumeration |
| `$EXPERT_LBA` | Deployment weight placement |
| Bitstream / ELF load steps | Lab-specific SDK GUI / xsct |

No host helper is required for the command path above.

---

## Static option audit

Used options only:

`--opcode --namespace-id --cdw3 --cdw10 --cdw11 --cdw12 --cdw13 --cdw14`
`--write --read --data-len --input-file --raw-binary`

No `--cdw4`.
