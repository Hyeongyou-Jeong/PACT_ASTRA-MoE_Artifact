//////////////////////////////////////////////////////////////////////////////////
// nvme_io_cmd.c for Cosmos+ OpenSSD
// Copyright (c) 2016 Hanyang University ENC Lab.
// Contributed by Yong Ho Song <yhsong@enc.hanyang.ac.kr>
//				  Youngjin Jo <yjjo@enc.hanyang.ac.kr>
//				  Sangjin Lee <sjlee@enc.hanyang.ac.kr>
//				  Jaewook Kwak <jwkwak@enc.hanyang.ac.kr>
//
// This file is part of Cosmos+ OpenSSD.
//
// Cosmos+ OpenSSD is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3, or (at your option)
// any later version.
//
// Cosmos+ OpenSSD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Cosmos+ OpenSSD; see the file COPYING.
// If not, see <http://www.gnu.org/licenses/>.
//////////////////////////////////////////////////////////////////////////////////

#include "xil_types.h"
#include "xil_printf.h"
#include "xtime_l.h"
#include "xparameters.h"

#include "stdio.h"
#include "debug.h"
#include "io_access.h"

#include "nvme.h"
#include "host_lld.h"
#include "nvme_io_cmd.h"
#include "../gemm/gemm_dispatcher.h"

#include "../ftl_config.h"
#include "../request_transform.h"
#include "../gemm/npu_driver.h"

/* Direct DMA max length enforced by set_direct_*_dma(). */
#define ACT_XFER_MAX_BYTES		0x1000u

static void ActReject(unsigned int cmdSlotTag)
{
	NVME_COMPLETION cpl;

	cpl.statusFieldWord = 0;
	cpl.statusField.SC = SC_INVALID_FIELD_IN_COMMAND;
	cpl.statusField.SCT = SCT_GENERIC_COMMAND_STATUS;
	set_auto_nvme_cpl(cmdSlotTag, 0, cpl.statusFieldWord);
}

/*
 * Host PRP <-> device DRAM using existing set_direct_{rx,tx}_dma
 * (same pattern as handle_identify). len must be 1..ACT_XFER_MAX_BYTES.
 * isHostToDev: 1 = ACT_WRITE (rx), 0 = ACT_READ (tx).
 */
static int ActDirectPrpXfer(NVME_IO_COMMAND *cmd, unsigned int devAddr,
		unsigned int len, unsigned int isHostToDev)
{
	unsigned int prp[2];
	unsigned int remain = len;
	unsigned int curDev = devAddr;
	unsigned int chunk;
	unsigned int pageRemain;
	unsigned int usedPrp2 = 0;

	if (len == 0 || len > ACT_XFER_MAX_BYTES)
		return -1;
	if ((cmd->PRP1[0] & 0x3) != 0)
		return -1;

	prp[0] = cmd->PRP1[0];
	prp[1] = cmd->PRP1[1];

	while (remain > 0) {
		pageRemain = 0x1000u - (prp[0] & 0xFFFu);
		chunk = remain;
		if (chunk > pageRemain)
			chunk = pageRemain;
		if (chunk > ACT_XFER_MAX_BYTES)
			chunk = ACT_XFER_MAX_BYTES;

		if (isHostToDev)
			set_direct_rx_dma(curDev, prp[1], prp[0], chunk);
		else
			set_direct_tx_dma(curDev, prp[1], prp[0], chunk);

		remain -= chunk;
		curDev += chunk;

		if (remain == 0)
			break;

		/* Next host page via PRP2 (identify-style; no PRP list walk). */
		if (usedPrp2 || (cmd->PRP2[0] & 0x3) != 0)
			return -1;
		prp[0] = cmd->PRP2[0];
		prp[1] = cmd->PRP2[1];
		usedPrp2 = 1;
	}

	if (isHostToDev)
		check_direct_rx_dma_done();
	else
		check_direct_tx_dma_done();

	return 0;
}

void handle_nvme_io_read(unsigned int cmdSlotTag, NVME_IO_COMMAND *nvmeIOCmd)
{
	IO_READ_COMMAND_DW12 readInfo12;
	unsigned int startLba[2];
	unsigned int nlb;

	readInfo12.dword = nvmeIOCmd->dword[12];

	startLba[0] = nvmeIOCmd->dword[10];
	startLba[1] = nvmeIOCmd->dword[11];
	nlb = readInfo12.NLB;

	ASSERT(startLba[0] < storageCapacity_L && (startLba[1] < STORAGE_CAPACITY_H || startLba[1] == 0));
	ASSERT((nvmeIOCmd->PRP1[0] & 0x3) == 0 && (nvmeIOCmd->PRP2[0] & 0x3) == 0);
	ASSERT(nvmeIOCmd->PRP1[1] < 0x10000 && nvmeIOCmd->PRP2[1] < 0x10000);

	ReqTransNvmeToSlice(cmdSlotTag, startLba[0], nlb, IO_NVM_READ);
}

void handle_nvme_io_write(unsigned int cmdSlotTag, NVME_IO_COMMAND *nvmeIOCmd)
{
	IO_WRITE_COMMAND_DW12 writeInfo12;
	unsigned int startLba[2];
	unsigned int nlb;

	writeInfo12.dword = nvmeIOCmd->dword[12];

	startLba[0] = nvmeIOCmd->dword[10];
	startLba[1] = nvmeIOCmd->dword[11];
	nlb = writeInfo12.NLB;

	ASSERT(startLba[0] < storageCapacity_L && (startLba[1] < STORAGE_CAPACITY_H || startLba[1] == 0));
	ASSERT((nvmeIOCmd->PRP1[0] & 0xF) == 0 && (nvmeIOCmd->PRP2[0] & 0xF) == 0);
	ASSERT(nvmeIOCmd->PRP1[1] < 0x10000 && nvmeIOCmd->PRP2[1] < 0x10000);

	ReqTransNvmeToSlice(cmdSlotTag, startLba[0], nlb, IO_NVM_WRITE);
}

/* Host -> IFMAP DRAM (no NAND). */
void handle_nvme_io_act_write(unsigned int cmdSlotTag, NVME_IO_COMMAND *nvmeIOCmd)
{
	IO_ACT_COMMAND_DW10 act10;
	IO_ACT_COMMAND_DW12 act12;
	unsigned int offset;
	unsigned int len;
	unsigned int dest;
	unsigned int ifmapEnd = OFMAP_BUFFER_BASE_ADDR;

	act10.dword = nvmeIOCmd->dword[10];
	act12.dword = nvmeIOCmd->dword[12];
	offset = act10.ActOffset;
	len = act12.ActLen;

	if (len == 0 || len > ACT_XFER_MAX_BYTES) {
		ActReject(cmdSlotTag);
		return;
	}
	if (offset > (ifmapEnd - IFMAP_BUFFER_BASE_ADDR)) {
		ActReject(cmdSlotTag);
		return;
	}
	if (len > (ifmapEnd - IFMAP_BUFFER_BASE_ADDR - offset)) {
		ActReject(cmdSlotTag);
		return;
	}

	dest = IFMAP_BUFFER_BASE_ADDR + offset;
	if (ActDirectPrpXfer(nvmeIOCmd, dest, len, 1) != 0) {
		ActReject(cmdSlotTag);
		return;
	}

	set_auto_nvme_cpl(cmdSlotTag, 0, 0);
}

/* OFMAP DRAM -> Host (no NAND). */
void handle_nvme_io_act_read(unsigned int cmdSlotTag, NVME_IO_COMMAND *nvmeIOCmd)
{
	IO_ACT_COMMAND_DW10 act10;
	IO_ACT_COMMAND_DW12 act12;
	unsigned int offset;
	unsigned int len;
	unsigned int src;
	unsigned int ofmapEnd = WEIGHT_BUFFER_BASE_ADDR;

	act10.dword = nvmeIOCmd->dword[10];
	act12.dword = nvmeIOCmd->dword[12];
	offset = act10.ActOffset;
	len = act12.ActLen;

	if (len == 0 || len > ACT_XFER_MAX_BYTES) {
		ActReject(cmdSlotTag);
		return;
	}
	if (offset > (ofmapEnd - OFMAP_BUFFER_BASE_ADDR)) {
		ActReject(cmdSlotTag);
		return;
	}
	if (len > (ofmapEnd - OFMAP_BUFFER_BASE_ADDR - offset)) {
		ActReject(cmdSlotTag);
		return;
	}

	src = OFMAP_BUFFER_BASE_ADDR + offset;
	if (ActDirectPrpXfer(nvmeIOCmd, src, len, 0) != 0) {
		ActReject(cmdSlotTag);
		return;
	}

	set_auto_nvme_cpl(cmdSlotTag, 0, 0);
}

void handle_nvme_io_gemm(unsigned int cmdSlotTag, NVME_IO_COMMAND *nvmeIOCmd)
{
	IO_GEMM_COMMAND_DW2 gemmInfo2;
	IO_GEMM_COMMAND_DW3 gemmInfo3;
	IO_GEMM_COMMAND_DW12 gemmInfo12;
	IO_GEMM_COMMAND_DW13 gemmInfo13;
	IO_GEMM_COMMAND_DW14 gemmInfo14;

	unsigned int startLba[2];
	unsigned int ifmapBufferAddr, ofmapBufferAddr;
	unsigned int weightLogicalAddr;
	unsigned int numRows, outCols;

	gemmInfo2.dword = nvmeIOCmd->dword[2];
	gemmInfo3.dword = nvmeIOCmd->dword[3];
	gemmInfo12.dword = nvmeIOCmd->dword[12];
	gemmInfo13.dword = nvmeIOCmd->dword[13];
	gemmInfo14.dword = nvmeIOCmd->dword[14];
	(void)gemmInfo2;

	startLba[0] = nvmeIOCmd->dword[10];
	startLba[1] = nvmeIOCmd->dword[11];

	ASSERT(startLba[0] < storageCapacity_L && (startLba[1] < STORAGE_CAPACITY_H || startLba[1] == 0));

	weightLogicalAddr = startLba[0];
	ifmapBufferAddr = IFMAP_BUFFER_BASE_ADDR + gemmInfo12.IfmapAddr;
	ofmapBufferAddr = OFMAP_BUFFER_BASE_ADDR + gemmInfo13.OfmapAddr;

	ASSERT((nvmeIOCmd->PRP1[0] & 0xF) == 0 && (nvmeIOCmd->PRP2[0] & 0xF) == 0);
	ASSERT(nvmeIOCmd->PRP1[1] < 0x10000 && nvmeIOCmd->PRP2[1] < 0x10000);

	/* NUMR=CDW3, DLEN=CDW14 (nvme-cli / Linux passthrough accessible). */
	numRows = gemmInfo3.NUMR;
	outCols = gemmInfo14.DLEN;

	gemm_dispatcher(cmdSlotTag, ifmapBufferAddr, ofmapBufferAddr, weightLogicalAddr, numRows, outCols);
}

void handle_nvme_io_cmd(NVME_COMMAND *nvmeCmd)
{
	NVME_IO_COMMAND *nvmeIOCmd;
	NVME_COMPLETION nvmeCPL;
	unsigned int opc;
	nvmeIOCmd = (NVME_IO_COMMAND*)nvmeCmd->cmdDword;

	opc = (unsigned int)nvmeIOCmd->OPC;

	switch(opc)
	{
		case IO_NVM_FLUSH:
		{
			xil_printf("IO Flush Command\r\n");
			nvmeCPL.dword[0] = 0;
			nvmeCPL.specific = 0x0;
			set_auto_nvme_cpl(nvmeCmd->cmdSlotTag, nvmeCPL.specific, nvmeCPL.statusFieldWord);
			break;
		}
		case IO_NVM_WRITE:
		{
			handle_nvme_io_write(nvmeCmd->cmdSlotTag, nvmeIOCmd);
			break;
		}
		case IO_NVM_READ:
		{
			handle_nvme_io_read(nvmeCmd->cmdSlotTag, nvmeIOCmd);
			break;
		}
		case IO_NVM_ACT_WRITE:
		{
			handle_nvme_io_act_write(nvmeCmd->cmdSlotTag, nvmeIOCmd);
			break;
		}
		case IO_NVM_ACT_READ:
		{
			handle_nvme_io_act_read(nvmeCmd->cmdSlotTag, nvmeIOCmd);
			break;
		}
		case IO_NVM_GEMM:
		{
			handle_nvme_io_gemm(nvmeCmd->cmdSlotTag, nvmeIOCmd);
			break;
		}
		default:
		{
			xil_printf("Not Support IO Command OPC: %X\r\n\r\n", opc);
			ASSERT(0);
			break;
		}
	}
}
