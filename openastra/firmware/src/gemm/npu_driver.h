/*
 * npu_driver.h
 *
 *  Created on: 2025.01.16.
 *      Author: hgjeong
 */
#include "xparameters.h"
#include "../memory_map.h"
#include "gemm_dispatcher.h"
#ifndef __NPU_DRIVER_H_
#define __NPU_DRIVER_H_

#ifdef	XPAR_NPU_BASEADDR
#define NPU_CONNECTED	1
#define NPU_BASEADDR	XPAR_NPU_BASEADDR

#define CSRA_DMA_CONTROL			((NPU_BASEADDR) + 0x48) //12
#define CSRA_NUM					((NPU_BASEADDR) + 0x50) //14

//DRAM -> SPM
#define CSRA_DRAM_SOURCE_ADDR		((NPU_BASEADDR) + 0x58) //16 DRAM Buffer Address
#define CSRA_SPM_DESTINATION_ADDR	((NPU_BASEADDR) + 0x60) //18 SPM Address

#define CSRA_DLA_STATE		 		((NPU_BASEADDR) + 0x68) //1a
#define CSRA_DLA_CONFIG_LOW 		((NPU_BASEADDR) + 0x70) //1c
#define CSRA_DLA_CONFIG_HIGH		((NPU_BASEADDR) + 0x78) //1e

#endif
// DRAM Buffer ???
#define IFMAP_BUFFER_BASE_ADDR		(ACTIVATION_BUFFER_BASE_ADDR) //16MB
#define OFMAP_BUFFER_BASE_ADDR		(IFMAP_BUFFER_BASE_ADDR + 0x01000000)
#define WMAP_BUFFER_BASE_ADDR		WEIGHT_BUFFER_BASE_ADDR

// NPU SRAM ???
#define DLA_WEIGHT_ADDR				0x00000000				//Weight SPM Address
#define DLA_ACTIVATION_ADDR			0x00004000				//SPM Address
#define DLA_PSUM_ADDR            	0x00000400				//Output Activation SPM Address

//default!!!  Native GEMM tile: 16x32 (4b) x 32x32 (4b) -> 16x32 (16b)
#define WEIGHT_SIZE              0x200		//32 x 32 (4bit) = 0x200
#define DLA_PSUM_SIZE            0x400		//16 x 32 (16bit) = 0x400
#define ACTIVATION_SIZE          0x100		//16 x 32 (4bit) = 0x100
//#define WEIGHT_SIZE              MOE_TENSOR_SIZE_IN_BYTE			//32 x 32 (4bit) = 0x200 !!HOST_WEIGHT_ADDR + WEIGHT_SIZE ?? Address ?? Output Activation ??
//#define DLA_PSUM_SIZE            OUTPUT_ACTIVATION_SIZE_IN_BYTE		//16 x 32 (16bit) = 0x400
//#define ACTIVATION_SIZE          INPUT_ACTIVATION_SIZE_IN_BYTE		//16 x 32 (4bit) = 0x100

// DMA ???? ???? ????
#define DMA_GO_BIT               0x80000000	   // 1000 0000 ..
#define DMA_ENABLE_MASK          0xE0000000    // 1110 0000 .. DRAM -> SPM dma control csr?? ???? ?? (Weight/Activation ???? ??)
#define DMA_WRITE_MODE           0xA0000000    // 1010 0000 .. SPM -> DRAM

// DLA ???? ??? ????? ???? bit ?????
#define DLA_DONE_MASK            0x1


/* Optional yield while polling DMA/NPU (TEP sets this to progress NAND). */
extern void (*npu_poll_yield)(void);

void checkDmaCompletion();
void checkNpuCompletion();
void DRAM2SPM_WeightMatrix();
void DRAM2SPM_InputActivationMatrix(unsigned int IfmapAddr);
void DRAM2SPM_WeightMatrix(unsigned int weightBufferAddr);
void SPM2NPU_InputActivationMatrix();
void NPU_Execution();
void SPM2DRAM_OutputActivation(unsigned int OfmapAddr);
void test();
void npu_test();
void fetch_npu_request(unsigned int weightBufferAddr, unsigned int IfmapAddr, unsigned int OfmapAddr, unsigned int TensorShape);
/* Same native sequence as fetch_npu_request without timing printf (TEP path). */
void fetch_npu_request_quiet(unsigned int weightBufferAddr, unsigned int IfmapAddr, unsigned int OfmapAddr);

#endif /* NPU_DRIVER_H_ */
