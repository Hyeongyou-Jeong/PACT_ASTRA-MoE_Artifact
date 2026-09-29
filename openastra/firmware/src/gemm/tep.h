/*
 * tep.h - Tile-based Expert Processing (TEP) for OpenASTRA FPGA prototype
 *
 * NPU is the FPGA prototype implementation of the paper's EPU.
 * One TEP tile == one native NPU-consumable weight block (see npu_driver.h).
 *
 * Native compute tile is a fixed-shape GEMM:
 *   Y[M x N_tile] = X[M x K] * W[K x N_tile]
 * Canonical NVMe opcode name is IO_NVM_GEMM (0x80).
 * DLEN (output columns N) is encoded in CDW14 for Linux nvme-cli.
 *
 * Weight layout contract (offline preparation; no runtime transpose):
 *   tile i occupies FLASH_PAGES_PER_TILE contiguous FTL slices starting at
 *   baseSlice + i * FLASH_PAGES_PER_TILE
 *   where baseSlice = expertBaseLba / NVME_BLOCKS_PER_SLICE
 *   (expertBaseLba must be slice-aligned).
 */

#ifndef __TEP_H_
#define __TEP_H_

#include "npu_driver.h"
#include "../ftl_config.h"

#ifndef TEP_DEBUG
#define TEP_DEBUG				0
#endif

#ifndef TEP_SOFTWARE_TEST
#define TEP_SOFTWARE_TEST		0
#endif

/*
 * Native NPU GEMM geometry (WEIGHT_SIZE / ACTIVATION_SIZE / DLA_PSUM_SIZE):
 *   ifmap  : M x K, 4-bit  -> ACTIVATION_SIZE
 *   weight : K x N, 4-bit  -> WEIGHT_SIZE
 *   ofmap  : M x N, 16-bit -> DLA_PSUM_SIZE
 * Column-wise tiling along N; no K-direction accumulation in v1.
 */
#define NPU_TILE_M				16
#define NPU_TILE_K				32
#define NPU_TILE_N				32

#define NPU_TILE_WEIGHT_BYTES	(WEIGHT_SIZE)		/* 0x200 */
#define NPU_TILE_IFMAP_BYTES	(ACTIVATION_SIZE)	/* 0x100 */
#define NPU_TILE_OFMAP_BYTES	(DLA_PSUM_SIZE)		/* 0x400 */

#define FLASH_PAGES_PER_TILE	1
#define TEP_WEIGHT_SLOT_BYTES	(BYTES_PER_DATA_REGION_OF_SLICE)
#define TEP_WEIGHT_SLOT_COUNT	(MOE_BUFFER_SLOT_CNT)

#define TEP_MAX_TILES			64
#define TEP_MAX_INFLIGHT_PAGES	(TEP_WEIGHT_SLOT_COUNT * FLASH_PAGES_PER_TILE)

#define TEP_REQ_NONE			0xffff
#define TEP_TILE_NONE			0xffff
#define TEP_SLOT_NONE			0xff

/* TepStartRequest return: 0 accepted (TEP owns cpl); <0 rejected (caller owns cpl). */
#define TEP_OK					0
#define TEP_ERR_BUSY			(-1)
#define TEP_ERR_ROWS			(-2)
#define TEP_ERR_COLS			(-3)
#define TEP_ERR_TILECNT			(-4)
#define TEP_ERR_OFMAP			(-5)
#define TEP_ERR_IFMAP			(-6)
#define TEP_ERR_LBA_ALIGN		(-7)

enum {
	TEP_TILE_FREE = 0,
	TEP_TILE_ALLOCATED,
	TEP_TILE_FLASH_ISSUED,
	TEP_TILE_FLASH_READY,
	TEP_TILE_NPU_RUNNING,
	TEP_TILE_DONE
};

enum {
	TEP_REQ_IDLE = 0,
	TEP_REQ_ACTIVE,
	TEP_REQ_ABORTING,
	TEP_REQ_COMPLETING
};

enum {
	TEP_SLOT_FREE = 0,
	TEP_SLOT_FILLING,
	TEP_SLOT_READY,
	TEP_SLOT_NPU_BUSY
};

typedef struct _TEP_TILE {
	unsigned short tileIndex;
	unsigned short state;
	unsigned short bufferSlot;
	unsigned short expectedPages;
	unsigned short completedPages;
	unsigned short pageSeen;
	unsigned int weightSliceBase;
	unsigned int ofmapAddr;
	unsigned int ifmapAddr;
} TEP_TILE;

typedef struct _TEP_REQUEST {
	unsigned short state;
	unsigned short cmdSlotTag;
	unsigned short tileCount;
	unsigned short completedTileCount;
	unsigned short nextIssueTile;
	unsigned short cplSent;
	unsigned short outstandingPageCount;
	unsigned short abortStatusWord;	/* NVMe statusFieldWord when aborting */
	unsigned int expertBaseLba;
	unsigned int ifmapBaseAddr;
	unsigned int ofmapBaseAddr;
	unsigned int numRows;
	unsigned int outCols;
	TEP_TILE tile[TEP_MAX_TILES];
} TEP_REQUEST;

typedef struct _TEP_SLOT {
	unsigned short state;
	unsigned short tileIndex;
	unsigned int dramAddr;
} TEP_SLOT;

typedef struct _TEP_INFLIGHT_PAGE {
	unsigned short valid;
	unsigned short reqSlotTag;
	unsigned short tileIndex;
	unsigned short pageIndex;
} TEP_INFLIGHT_PAGE;

void InitTep(void);
int TepIsActive(void);		/* ACTIVE, ABORTING, or COMPLETING */
void TepService(void);

/*
 * Returns TEP_OK if accepted (TEP owns success/error completion).
 * Returns <0 if rejected before ACTIVE (caller must send immediate error cpl).
 */
int TepStartRequest(unsigned int cmdSlotTag,
		unsigned int ifmapAddr,
		unsigned int ofmapAddr,
		unsigned int expertBaseLba,
		unsigned int numRows,
		unsigned int outCols);

void TepNotifyPageCompletion(unsigned int reqSlotTag);

#if TEP_SOFTWARE_TEST
void TepRunSoftwareTests(void);
#endif

#endif /* __TEP_H_ */
