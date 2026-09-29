/*
 * gemm_dispatcher.c
 *
 *  Created on: 2024. 12. 26.
 *      Author: hgjeong
 */

#include "xil_types.h"
#include "xil_printf.h"
#include "xtime_l.h"
#include "xparameters.h"
#include "stdio.h"
#include "xil_exception.h"
#include "xil_printf.h"
#include "stdint.h"

#include "../nvme/debug.h"
#include "../nvme/io_access.h"
#include "math.h"
#include "limits.h"

//#include "../nvme/nvme.h"
#include "gemm_dispatcher.h"
#include "../nvme/host_lld.h"
#include "../memory_map.h"
#include "npu_driver.h"
#include "tep.h"
#include "../nvme/nvme.h"
#include "../request_format.h"
#include "../request_allocation.h"
#include "../request_schedule.h"
#include "../address_translation.h"
#include "../ftl_config.h"

HOST_REQUEST_FOR_GEMM* gemmNandTable;
BUFFER_INFO_FOR_GEMM* gemmBufferTable;

void InitGemmTable()
{
	InitGemmCompletionTable();
	InitGemmBufferTable();
	InitTep();
}

void InitGemmCompletionTable()
{
	gemmNandTable = (HOST_REQUEST_FOR_GEMM*)HOST_REQUEST_FOR_GEMM_ADDR;

	for(int i = 0; i < TOTAL_GEMM_HOST_REQ ; i++)
	{
		for(int j = 0; j < TOTAL_GEMM_NAND_REQ; j++)
		{
			gemmNandTable->hostReq[i].nandReq[j] = NAND_REQ_FOR_GEMM_NONE;
		}
		gemmNandTable->hostReq[i].reqCnt = 0;
	}
}

void InitGemmBufferTable()
{
	gemmBufferTable = (BUFFER_INFO_FOR_GEMM*)GEMM_BUFFER_TABLE_ADDR;
	gemmBufferTable->bufferSlotIdleCnt = MOE_BUFFER_SLOT_CNT;
	for(int i = 0; i < MOE_BUFFER_SLOT_CNT ; i++)
	{
		gemmBufferTable->bufferSlot[i].bufferState = BUFFER_STATE_EMPTY;
		gemmBufferTable->bufferSlot[i].weightLBA = 0;
		gemmBufferTable->bufferSlot[i].usageTime = 0;
		gemmBufferTable->bufferSlot[i].cmdSlotTag = 0;
	}

}

void gemm_dispatcher(unsigned int cmdSlotTag, unsigned int ifmapBufAddr, unsigned int ofmapBufAddr,
		unsigned int weightLogicalAddr, unsigned int numRows, unsigned int outCols)
{
	int rc;
	NVME_COMPLETION cpl;

	/*
	 * TEP owns NVMe completion once TepStartRequest returns TEP_OK
	 * (success or async abort after partial issue).
	 * On reject (<0), caller/dispatcher owns the immediate error cpl.
	 */
	rc = TepStartRequest(cmdSlotTag, ifmapBufAddr, ofmapBufAddr, weightLogicalAddr, numRows, outCols);
	if (rc != TEP_OK) {
		xil_printf("[TEP] reject GEMM cmdSlot=%u rc=%d NUMR=%u DLEN=%u\r\n",
			cmdSlotTag, rc, numRows, outCols);
		cpl.statusFieldWord = 0;
		cpl.statusField.SC = SC_INVALID_FIELD_IN_COMMAND;
		cpl.statusField.SCT = SCT_GENERIC_COMMAND_STATUS;
		set_auto_nvme_cpl(cmdSlotTag, 0, cpl.statusFieldWord);
	}
}


Weight_Buffer buffer_slot_manager(unsigned int weightAddr)
{
    for (int i = 0; i < MOE_BUFFER_SLOT_CNT; i++) {
    	gemmBufferTable->bufferSlot[i].usageTime++; // ??? ?????? ??? ???? ????
    }

    Weight_Buffer weightBuffer;
    int buffer_slot_tag;
    int lruIndex = -1;

    // Buffer Hit!!
    for (buffer_slot_tag = 0; buffer_slot_tag < MOE_BUFFER_SLOT_CNT; buffer_slot_tag++) {
        if ((gemmBufferTable->bufferSlot[buffer_slot_tag].bufferState != BUFFER_STATE_EMPTY) && (gemmBufferTable->bufferSlot[buffer_slot_tag].weightLBA == weightAddr)) {
        	gemmBufferTable->bufferSlot[buffer_slot_tag].usageTime = 0; // ??? ???? ????
        	gemmBufferTable->bufferSlot[buffer_slot_tag].bufferState = BUFFER_STATE_BUSY;
        	gemmBufferTable->bufferSlotIdleCnt--;
            weightBuffer.address = WEIGHT_BUFFER_BASE_ADDR + buffer_slot_tag * MOE_TENSOR_SIZE_IN_BYTE;
            weightBuffer.hit = 1;
            weightBuffer.buffer_slot_tag = buffer_slot_tag;
            return weightBuffer;
        }
    }

    // Buffer Miss!! -> LRU Buffer Management
    unsigned int lruTime = 0;
    for (buffer_slot_tag = 0; buffer_slot_tag < MOE_BUFFER_SLOT_CNT; buffer_slot_tag++) {
        if (gemmBufferTable->bufferSlot[buffer_slot_tag].bufferState == 0) {
        	gemmBufferTable->bufferSlot[buffer_slot_tag].weightLBA = weightAddr;
        	gemmBufferTable->bufferSlot[buffer_slot_tag].usageTime = 0; // ???? ???? ?????? ???? ????
        	gemmBufferTable->bufferSlot[buffer_slot_tag].bufferState = BUFFER_STATE_BUSY;
        	gemmBufferTable->bufferSlotIdleCnt--;
            weightBuffer.address = WEIGHT_BUFFER_BASE_ADDR + buffer_slot_tag * MOE_TENSOR_SIZE_IN_BYTE;
            weightBuffer.hit = 0;
            weightBuffer.buffer_slot_tag = buffer_slot_tag;
            return weightBuffer;
        }
        // LRU ???? ???
        if (gemmBufferTable->bufferSlot[buffer_slot_tag].usageTime > lruTime) {
        	lruTime = gemmBufferTable->bufferSlot[buffer_slot_tag].usageTime;
            lruIndex = buffer_slot_tag;
        }
    }
    buffer_slot_tag = lruIndex;
    // LRU ???? ???
    gemmBufferTable->bufferSlot[buffer_slot_tag].weightLBA = weightAddr;   // ?????? weight ???? ???
    gemmBufferTable->bufferSlot[buffer_slot_tag].usageTime = 0;            // ??? ???? ????
	gemmBufferTable->bufferSlot[buffer_slot_tag].bufferState = BUFFER_STATE_BUSY;
	gemmBufferTable->bufferSlotIdleCnt--;

	weightBuffer.address = WEIGHT_BUFFER_BASE_ADDR + lruIndex * MOE_TENSOR_SIZE_IN_BYTE;
	weightBuffer.hit = 0;
    weightBuffer.buffer_slot_tag = buffer_slot_tag;
    return weightBuffer;
}

//???? ???
Tensor llm_data_table(unsigned int weightAddr)
{
	//int logicalAddr;
	//int tensorShape;
	Tensor tensor;

	tensor.x_axis = 1024;
	tensor.y_axis = 1024;
	tensor.precision = 16;

	return tensor;
}

void fetch_read_request(unsigned int cmdSlotTag, unsigned int weightAddr, unsigned int weightSizeInByte, unsigned int weightBufferAddr)
{
	unsigned int reqSlotTag;
	unsigned int weightLBA = weightAddr / NVME_BLOCKS_PER_SLICE;
	unsigned int flash_read_request_cnt = weightSizeInByte / BYTES_PER_DATA_REGION_OF_NAND_ROW;
	if (weightSizeInByte % BYTES_PER_DATA_REGION_OF_NAND_ROW != 0)
		flash_read_request_cnt++;
	//flash_read_request_cnt = ceilf(c);
	for(int addrOffset = 0; addrOffset < flash_read_request_cnt; addrOffset++)
	{
		unsigned int PPA = logicalSliceMapPtr->logicalSlice[weightLBA].virtualSliceAddr;
		reqSlotTag = GetFromFreeReqQ();

		reqPoolPtr->reqPool[reqSlotTag].reqType = REQ_TYPE_NAND;
		reqPoolPtr->reqPool[reqSlotTag].reqCode = REQ_CODE_READ;
		reqPoolPtr->reqPool[reqSlotTag].nvmeCmdSlotTag = cmdSlotTag;
		reqPoolPtr->reqPool[reqSlotTag].logicalSliceAddr = weightLBA + addrOffset;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.dataBufFormat = REQ_OPT_DATA_BUF_ADDR; // REQ_OPT_DATA_BUF_ADDR ?? ???
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.nandAddr = REQ_OPT_NAND_ADDR_VSA;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.nandEcc = REQ_OPT_NAND_ECC_ON;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.nandEccWarning = REQ_OPT_NAND_ECC_WARNING_OFF;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.rowAddrDependencyCheck = REQ_OPT_ROW_ADDR_DEPENDENCY_CHECK;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.blockSpace = REQ_OPT_BLOCK_SPACE_MAIN;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.gemmCmd = 1;
		reqPoolPtr->reqPool[reqSlotTag].dataBufInfo.addr = weightBufferAddr; //??? ????? entry ????
		reqPoolPtr->reqPool[reqSlotTag].nandInfo.virtualSliceAddr = PPA; // LBA -> PPA

		SelectLowLevelReqQ(reqSlotTag);

		PushToGemmNandRequestTable(reqSlotTag);
	}

	//xil_printf("Enqueue GemmReq %d\r\n", cmdSlotTag);

	SyncAllLowLevelReqDone();
}

void PushToGemmNandRequestTable(unsigned int reqSlotTag)
{
	unsigned int hostReq = reqPoolPtr->reqPool[reqSlotTag].nvmeCmdSlotTag;
	unsigned int cnt = gemmNandTable->hostReq[hostReq].reqCnt;
	gemmNandTable->hostReq[hostReq].nandReq[cnt] = reqSlotTag;
	gemmNandTable->hostReq[hostReq].reqCnt++;
}

unsigned int CompletionGemmNandRequest(unsigned int reqSlotTag)
{
	unsigned int hostReq = reqPoolPtr->reqPool[reqSlotTag].nvmeCmdSlotTag;
	const unsigned int cnt = gemmNandTable->hostReq[hostReq].reqCnt;
	unsigned int returnValue = NAND_REQ_FOR_GEMM_RUNNING;

	if(reqPoolPtr->reqPool[reqSlotTag].reqOpt.gemmCmd == 0)
		return 0;

	/* TEP page-level notify while req metadata is still valid. */
	TepNotifyPageCompletion(reqSlotTag);

	if(cnt == 0)
	{
		xil_printf("There is no Gemm Nand requests\r\n");
		ASSERT(0);
	}

	for(int iter = cnt; iter >= 0; iter--)
	{
		if(reqSlotTag == gemmNandTable->hostReq[hostReq].nandReq[iter])
		{
			gemmNandTable->hostReq[hostReq].nandReq[iter] = NAND_REQ_FOR_GEMM_NONE;
			reqPoolPtr->reqPool[reqSlotTag].reqOpt.gemmCmd = 0;
/*
			for(int j = iter; j < cnt; j++)
			{
				gemmNandTable->hostReq[hostReq].nandReq[j] = gemmNandTable->hostReq[hostReq].nandReq[j+1];
			}
*/
			gemmNandTable->hostReq[hostReq].reqCnt--;
			break;
		}
	}

	if(gemmNandTable->hostReq[hostReq].reqCnt == 0)
		returnValue = NAND_REQ_FOR_GEMM_DONE;

	return returnValue;
}
