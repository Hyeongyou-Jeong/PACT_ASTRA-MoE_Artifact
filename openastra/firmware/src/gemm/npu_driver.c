/* npu_driver.c
 *
 *  Created on: 2025. 1. 16.
 *      Author: hgjeong
 */
#include "stdio.h"
#include "xil_exception.h"
#include "xil_printf.h"
#include "../nvme/debug.h"
#include "../nvme/io_access.h"
#include "math.h"
#include "limits.h"

#include "../memory_map.h"
#include "npu_driver.h"

#include "xil_types.h"
#include "xtime_l.h"
#include "xparameters.h"

void (*npu_poll_yield)(void) = 0;

// DMA GO bit clear wait (optional yield for TEP flash overlap)
void checkDmaCompletion() {
    while ((IO_READ32(CSRA_NUM) & DMA_GO_BIT) != 0) {
		if (npu_poll_yield)
			npu_poll_yield();
	}
}

void checkNpuCompletion() {
    while ((IO_READ32(CSRA_DLA_STATE) & DLA_DONE_MASK) == 0) {
		if (npu_poll_yield)
			npu_poll_yield();
	}
}

// DMA_ACTIVATION_SRAM_FILL: HOST_ACTIVATION_ADDR -> DLA_ACTIVATION_ADDR (DMA)
void DRAM2SPM_InputActivationMatrix(unsigned int IfmapAddr)
{
	IO_WRITE32(CSRA_DRAM_SOURCE_ADDR, IfmapAddr);
	IO_WRITE32(CSRA_SPM_DESTINATION_ADDR, DLA_ACTIVATION_ADDR);
	IO_WRITE32(CSRA_DMA_CONTROL, DMA_ENABLE_MASK);

	IO_WRITE32(CSRA_NUM, (ACTIVATION_SIZE + DMA_GO_BIT));

	checkDmaCompletion(); //Pooling
	IO_WRITE32(CSRA_DMA_CONTROL, 0);
}

//DMA_Weight_SRAM_FILL: HOST_WEIGHT_ADDR -> DLA_WEIGHT_ADDR (DMA)
void DRAM2SPM_WeightMatrix(unsigned int weightBufferAddr)
{
	IO_WRITE32(CSRA_DRAM_SOURCE_ADDR, weightBufferAddr);
	IO_WRITE32(CSRA_SPM_DESTINATION_ADDR, DLA_WEIGHT_ADDR);
	IO_WRITE32(CSRA_DMA_CONTROL, DMA_ENABLE_MASK);

	IO_WRITE32(CSRA_NUM, (WEIGHT_SIZE + DMA_GO_BIT));

    checkDmaCompletion(); //Pooling

}
//NPU Bug 20250207!! SPM Address 0x40~0x80, 0xc0~0x100 �?� �?�
/*
void DRAM2SPM_WeightMatrix(unsigned int weightBufferAddr)
{
	IO_WRITE32(CSRA_DRAM_SOURCE_ADDR, weightBufferAddr + 0x40);
	IO_WRITE32(CSRA_SPM_DESTINATION_ADDR, DLA_WEIGHT_ADDR + 0x40);
	IO_WRITE32(CSRA_DMA_CONTROL, DMA_ENABLE_MASK);

	IO_WRITE32(CSRA_NUM, (0x40 + DMA_GO_BIT));

    checkDmaCompletion(); //Pooling

	IO_WRITE32(CSRA_DRAM_SOURCE_ADDR, weightBufferAddr + 0xC0);
	IO_WRITE32(CSRA_SPM_DESTINATION_ADDR, DLA_WEIGHT_ADDR + 0xC0);
	IO_WRITE32(CSRA_DMA_CONTROL, DMA_ENABLE_MASK);

	IO_WRITE32(CSRA_NUM, (0x40 + DMA_GO_BIT));

    checkDmaCompletion(); //Pooling
}
*/
void SPM2NPU_WeightMatrix()
{
    IO_WRITE32(CSRA_DMA_CONTROL, 0);
    IO_WRITE32(CSRA_DLA_STATE, 0);
    IO_WRITE32(CSRA_DLA_CONFIG_LOW, 0x00000011);
    IO_WRITE32(CSRA_DLA_CONFIG_HIGH, 0x00240001);
    IO_WRITE32(CSRA_DLA_STATE, DMA_GO_BIT);

    checkNpuCompletion();	//Pooling
}

// L0_loading
void SPM2NPU_InputActivationMatrix()
{
    IO_WRITE32(CSRA_DMA_CONTROL, 0);
    IO_WRITE32(CSRA_DLA_STATE, 0);
    IO_WRITE32(CSRA_DLA_CONFIG_LOW, 0x00000f02);
    IO_WRITE32(CSRA_DLA_CONFIG_HIGH, 0x0012000f);
    IO_WRITE32(CSRA_DLA_STATE, DMA_GO_BIT);

    checkNpuCompletion();	//Pooling
}

// Core Execution
void NPU_Execution()
{
    IO_WRITE32(CSRA_DMA_CONTROL, 0);
    IO_WRITE32(CSRA_DLA_STATE, 0);
    IO_WRITE32(CSRA_DLA_CONFIG_LOW, 0x00001003);
    IO_WRITE32(CSRA_DLA_CONFIG_HIGH, 0x003B0010);
    IO_WRITE32(CSRA_DLA_STATE, DMA_GO_BIT);

    checkNpuCompletion();	//Pooling
}



// Core Execution
void NPU_Accumulation()
{
    IO_WRITE32(CSRA_DMA_CONTROL, 0);
    IO_WRITE32(CSRA_DLA_STATE, 0);
    IO_WRITE32(CSRA_DLA_CONFIG_LOW, 0x00081004);
    IO_WRITE32(CSRA_DLA_CONFIG_HIGH, 0x001C0001);
    IO_WRITE32(CSRA_DLA_STATE, DMA_GO_BIT);

    checkNpuCompletion();	//Pooling
}


// Result Load �?�: DLA_PSUM (DMA)
void SPM2DRAM_OutputActivation(unsigned int OfmapAddr)
{
    IO_WRITE32(CSRA_DLA_STATE, 0);
    //uint32_t src_addr = OFMAP_BUFFER_BASE_ADDR;
    IO_WRITE32(CSRA_DRAM_SOURCE_ADDR, OfmapAddr);
    IO_WRITE32(CSRA_SPM_DESTINATION_ADDR, DLA_PSUM_ADDR);
    IO_WRITE32(CSRA_DMA_CONTROL, DMA_WRITE_MODE);
    IO_WRITE32(CSRA_NUM, (DLA_PSUM_SIZE + DMA_GO_BIT));

    checkDmaCompletion();	//Pooling
    IO_WRITE32(CSRA_DMA_CONTROL, 0);
}

// NPUTEST
void npu_test()
{

	for (unsigned int addr = IFMAP_BUFFER_BASE_ADDR; addr < (IFMAP_BUFFER_BASE_ADDR + ACTIVATION_SIZE); addr += 2)
	{
		uint8_t a1, a2, a3, a4;
		a1 = (rand() % 8);
		a2 = (rand() % 8);
		a3 = (rand() % 8);
		a4 = (rand() % 8);

		uint16_t combined = PACK_FOUR_4BIT(a1, a2, a3, a4);
		IO_WRITE16(addr, combined);;
		printf("Input Activation:      %d\r\n", a1);
		printf("Input Activation:      %d\r\n", a2);
		printf("Input Activation:      %d\r\n", a3);
		printf("Input Activation:      %d\r\n", a4);
	}

	for (unsigned int addr = WEIGHT_BUFFER_BASE_ADDR; addr < (WEIGHT_BUFFER_BASE_ADDR + WEIGHT_SIZE); addr += 2)
	{
		uint8_t w1, w2, w3, w4;
		w1 = (rand() % 8);
		w2 = (rand() % 8);
		w3 = (rand() % 8);
		w4 = (rand() % 8);

		uint16_t combined = PACK_FOUR_4BIT(w1, w2, w3, w4);
		IO_WRITE16(addr, combined);;
		printf("Input Weight:      %d\r\n", w1);
		printf("Input Weight:      %d\r\n", w2);
		printf("Input Weight:      %d\r\n", w3);
		printf("Input Weight:      %d\r\n", w4);
	}


    DRAM2SPM_InputActivationMatrix(IFMAP_BUFFER_BASE_ADDR);
    DRAM2SPM_WeightMatrix(WMAP_BUFFER_BASE_ADDR);
    SPM2NPU_InputActivationMatrix();
    SPM2NPU_WeightMatrix();
    NPU_Execution();
    SPM2DRAM_OutputActivation(OFMAP_BUFFER_BASE_ADDR);
	for (unsigned int addr = OFMAP_BUFFER_BASE_ADDR; addr < (OFMAP_BUFFER_BASE_ADDR + DLA_PSUM_SIZE); addr += 2)
	{
		int16_t outputActivation = IO_READ16(addr);
		printf("Output Activation: %ld\r\n", outputActivation);
	}

}

void fetch_npu_request_quiet(unsigned int weightBufferAddr, unsigned int IfmapAddr, unsigned int OfmapAddr)
{
	DRAM2SPM_WeightMatrix(weightBufferAddr);
	DRAM2SPM_InputActivationMatrix(IfmapAddr);
	SPM2NPU_WeightMatrix();
	SPM2NPU_InputActivationMatrix();
	NPU_Execution();
	SPM2DRAM_OutputActivation(OfmapAddr);
}

void fetch_npu_request(unsigned int weightBufferAddr, unsigned int IfmapAddr, unsigned int OfmapAddr, unsigned int TensorShape)
{
	XTime tStart, tEnd;
	double elapsedTime;
	(void)TensorShape;

	XTime_GetTime(&tStart);
	DRAM2SPM_WeightMatrix(weightBufferAddr);
	XTime_GetTime(&tEnd);
	elapsedTime = (2 * (double)(tEnd - tStart) / (double)XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ) * 1000;
	printf("DRAM2SPM_WeightMatrix execution time: %f ms\r\n", elapsedTime);

	XTime_GetTime(&tStart);
	DRAM2SPM_InputActivationMatrix(IfmapAddr);
	XTime_GetTime(&tEnd);
	elapsedTime = (2 * (double)(tEnd - tStart) / (double)XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ) * 1000;
	printf("DRAM2SPM_InputActivationMatrix execution time: %f ms\r\n", elapsedTime);

	XTime_GetTime(&tStart);
	SPM2NPU_WeightMatrix();
	XTime_GetTime(&tEnd);
	elapsedTime = (2 * (double)(tEnd - tStart) / (double)XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ) * 1000;
	printf("SPM2NPU_WeightMatrix execution time: %f ms\r\n", elapsedTime);

	XTime_GetTime(&tStart);
	SPM2NPU_InputActivationMatrix();
	XTime_GetTime(&tEnd);
	elapsedTime = (2 * (double)(tEnd - tStart) / (double)XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ) * 1000;
	printf("SPM2NPU_InputActivationMatrix execution time: %f ms\r\n", elapsedTime);

	XTime_GetTime(&tStart);
	NPU_Execution();
	XTime_GetTime(&tEnd);
	elapsedTime = (2 * (double)(tEnd - tStart) / (double)XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ) * 1000;
	printf("NPU_Execution execution time: %f ms\r\n", elapsedTime);

	XTime_GetTime(&tStart);
	SPM2DRAM_OutputActivation(OfmapAddr);
	XTime_GetTime(&tEnd);
	elapsedTime = (2 * (double)(tEnd - tStart) / (double)XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ) * 1000;
	printf("SPM2DRAM_OutputActivation execution time: %f ms\r\n", elapsedTime);
}
