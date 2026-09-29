/*
 * gemm_dispatcher.h
 *
 *  Created on: 2024. 12. 26.
 *      Author: hgjeong
 */

#ifndef __GEMM_DISPATCHER_H_
#define __GEMM_DISPATCHER_H_

// ACTIVATION_BUFFER_BASE_ADDR		0x01000000 //16MB
//IFMAP_BUFFER_BASE_ADDR			0x02000000
// OFMAP_BUFFER_BASE_ADDR			0x03000000
// WEIGHT_BUFFER_BASE_ADDR			0x04000000 // ~0x10000000

#define INPUT_ACTIVATION_PRECISION		(4)	//bit
#define INPUT_ACTIVATION_X_AXIS			(1024)
#define INPUT_ACTIVATION_Y_AXIS			(1)
#define INPUT_ACTIVATION_SIZE_IN_BYTE	((INPUT_ACTIVATION_X_AXIS * INPUT_ACTIVATION_X_AXIS * INPUT_ACTIVATION_Y_AXIS) / 8)

#define OUTPUT_ACTIVATION_PRECISION		(16) //bit
#define OUTPUT_ACTIVATION_X_AXIS		(1024)
#define OUTPUT_ACTIVATION_Y_AXIS		(1)
#define OUTPUT_ACTIVATION_SIZE_IN_BYTE	((OUTPUT_ACTIVATION_X_AXIS * OUTPUT_ACTIVATION_X_AXIS * OUTPUT_ACTIVATION_Y_AXIS) / 8)

#define MOE_PRECISION					(4) //bit
#define MOE_TENSOR_X_AXIS				(2048)
#define MOE_TENSOR_Y_AXIS				(2048)
#define MOE_TENSOR_SIZE_IN_BYTE			((MOE_TENSOR_X_AXIS * MOE_TENSOR_Y_AXIS * MOE_PRECISION) / 8) //2MB
#define MOE_TENSOR_SIZE_IN_MBYTE		(MOE_TENSOR_SIZE_IN_BYTE / 1024 / 1024)
#define MOE_BUFFER_SIZE_IN_MBYTE		(4)
//#define MOE_BUFFER_SLOT_CNT				(MOE_BUFFER_SIZE_IN_MBYTE * 1024 * 1024) / (MOE_TENSOR_SIZE_IN_BYTE)
#define MOE_BUFFER_SLOT_CNT				2

//Buffer State
#define BUFFER_STATE_EMPTY			0
#define BUFFER_STATE_READY			1
#define BUFFER_STATE_BUSY			2

#define NAND_REQ_FOR_GEMM_NONE		(0xffff)
#define NAND_REQ_FOR_GEMM_RUNNING 	(0x0)
#define NAND_REQ_FOR_GEMM_DONE 		(0x1)
#define TOTAL_GEMM_HOST_REQ			(1024)
#define TOTAL_GEMM_NAND_REQ			(128) //64pages per 1 megabyte host request

typedef struct _Tensor
{
    unsigned int x_axis;
    unsigned int y_axis;
    unsigned int precision;
} Tensor;

typedef struct _Weight_Buffer
{
    unsigned int address;
    unsigned int hit;
    unsigned int buffer_slot_tag;
} Weight_Buffer;

typedef struct _NAND_COMPLETION_TABLE_FOR_GEMM
{
	unsigned int nandReq[TOTAL_GEMM_NAND_REQ];
	unsigned int reqCnt : 8;
	unsigned int cmdValid : 1;
	unsigned int cmdSlotTag : 10;
	unsigned int reserved : 17;
} NAND_COMPLETION_TABLE_FOR_GEMM, *P_NAND_COMPLETION_TABLE_FOR_GEMM;

typedef struct _HOST_REQUEST_FOR_GEMM
{
	NAND_COMPLETION_TABLE_FOR_GEMM hostReq[TOTAL_GEMM_HOST_REQ];
} HOST_REQUEST_FOR_GEMM;

typedef struct _BUFFER_TABLE_FOR_GEMM
{
	unsigned int bufferState : 3;
	unsigned int weightLBA;
	unsigned int cmdSlotTag : 10;
	unsigned int usageTime : 8;
	//unsigned int reserved : 11;
} BUFFER_TABLE_FOR_GEMM;

typedef struct _BUFFER_INFO_FOR_GEMM
{
	BUFFER_TABLE_FOR_GEMM bufferSlot[MOE_BUFFER_SLOT_CNT];
	unsigned int bufferSlotIdleCnt;
} BUFFER_INFO_FOR_GEMM, *P_BUFFER_INFO_FOR_GEMM;

void InitGemmTable();
void InitGemmCompletionTable();
void InitGemmBufferTable();
/* MoE/TEP entry via GEMM dispatcher — now starts async TEP (see tep.c). */
void gemm_dispatcher(unsigned int cmdSlotTag, unsigned int ifmapBufAddr, unsigned int ofmapBufAddr,
		unsigned int weightLogicalAddr, unsigned int numRows, unsigned int outCols);
Weight_Buffer buffer_slot_manager(unsigned int weightAddr);
Tensor llm_data_table(unsigned int weightAddr);
void fetch_read_request(unsigned int cmdSlotTag, unsigned int weightAddr, unsigned int weightSizeInByte, unsigned int weightBufferAddr);
void PushToGemmNandRequestTable(unsigned int reqSlotTag);
unsigned int CompletionGemmNandRequest(unsigned int reqSlotTag);

extern HOST_REQUEST_FOR_GEMM* gemmNandTable;
extern BUFFER_INFO_FOR_GEMM* gemmBufferTable;

#endif
