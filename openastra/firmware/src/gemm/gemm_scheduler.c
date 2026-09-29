/*
 * gemm_scheduler.c
 *
 *  Created on: 2024. 12. 17.
 *      Author: hgjeong
 */
#include "stdio.h"
#include "xil_exception.h"
#include "xil_printf.h"
#include "../nvme/debug.h"
#include "../nvme/io_access.h"

#include "../nvme/nvme.h"
#include "gemm_scheduler.h"
#include "../nvme/host_lld.h"

void reset_gemm_scheduler_queue(SCHEDULER_COMMAND *gemm_scheduler_queue) {
    for (int i = 0; i < SCHEDULER_WINDOW; i++) {
    	gemm_scheduler_queue[i].cmdValid = 0;
    	gemm_scheduler_queue[i].qOrder = 0;
    	gemm_scheduler_queue[i].qID = 0;
    	gemm_scheduler_queue[i].cmdSlotTag = 0;
    	gemm_scheduler_queue[i].cmdSeqNum = 0;
    	gemm_scheduler_queue[i].cmdDword[16] = 0;
    }
}

void enqueue_gemm_scheduler(SCHEDULER_COMMAND *gemm_scheduler_queue, NVME_COMMAND *nvmeCmd, int request_queue_cnt)
{
	int request_queue_index;
    for (request_queue_index = 0; request_queue_index < SCHEDULER_WINDOW; request_queue_index++) {
    	if (gemm_scheduler_queue[request_queue_index].cmdValid == 0)
    		break;
    }
    gemm_scheduler_queue[request_queue_index].cmdValid = 1;
	gemm_scheduler_queue[request_queue_index].qOrder = request_queue_cnt;
	gemm_scheduler_queue[request_queue_index].qID = nvmeCmd->qID;
	gemm_scheduler_queue[request_queue_index].cmdSlotTag = nvmeCmd->cmdSlotTag;
	gemm_scheduler_queue[request_queue_index].cmdSeqNum = nvmeCmd->cmdSeqNum;
    for (int i = 0; i < 16; i++) {
        gemm_scheduler_queue[request_queue_index].cmdDword[i] = nvmeCmd->cmdDword[i];
    }
}

void fetch_request_from_scheduler_queue(SCHEDULER_COMMAND *gemm_scheduler_queue, NVME_COMMAND *nvmeCmd)
{
	int request_queue_index;
	static uint64_t recentAddress = 0;
	static uint64_t currentAddress = 0;
	if (SCHEDULING_ALGORITHM == 0) //FIFO
	{
		for (int i = 0; i < SCHEDULER_WINDOW; i++) {
			if ((gemm_scheduler_queue[i].cmdValid == 1) && (gemm_scheduler_queue[i].qOrder == 0))
				request_queue_index = i;
			gemm_scheduler_queue[i].qOrder--;
		}
		gemm_scheduler_queue[request_queue_index].cmdValid = 0;
		nvmeCmd->qID = gemm_scheduler_queue[request_queue_index].qID;
		nvmeCmd->cmdSlotTag = gemm_scheduler_queue[request_queue_index].cmdSlotTag;
		nvmeCmd->cmdSeqNum = gemm_scheduler_queue[request_queue_index].cmdSeqNum;
		for (int i = 0; i < 16; i++) {
			nvmeCmd->cmdDword[i] = gemm_scheduler_queue[request_queue_index].cmdDword[i];
		}
	} //FIFO(end)
	else if (SCHEDULING_ALGORITHM == 1) //MoE Rescheduling
	{

		for (int i = 0; i < SCHEDULER_WINDOW; i++) {
			currentAddress = ((uint64_t)gemm_scheduler_queue[i].cmdDword[11] << 32) | gemm_scheduler_queue[i].cmdDword[10];
			if ((gemm_scheduler_queue[i].cmdValid == 1) && (currentAddress == recentAddress))
			{
				request_queue_index = i;
				for (int j = 0; j < SCHEDULER_WINDOW; j++) {
					if (gemm_scheduler_queue[j].qOrder > gemm_scheduler_queue[request_queue_index].qOrder)
						gemm_scheduler_queue[j].qOrder--;
				}
				break;
			}

			if (i == SCHEDULER_WINDOW - 1)
			{
				for (int k = 0; k < SCHEDULER_WINDOW; k++) {
					if ((gemm_scheduler_queue[k].cmdValid == 1) && (gemm_scheduler_queue[k].qOrder == 0))
					{
						request_queue_index = k;
						currentAddress =  ((uint64_t)gemm_scheduler_queue[k].cmdDword[11] << 32) | gemm_scheduler_queue[k].cmdDword[10];
					}
					gemm_scheduler_queue[k].qOrder--;
				}
			}
		}
		gemm_scheduler_queue[request_queue_index].cmdValid = 0;
		nvmeCmd->qID = gemm_scheduler_queue[request_queue_index].qID;
		nvmeCmd->cmdSlotTag = gemm_scheduler_queue[request_queue_index].cmdSlotTag;
		nvmeCmd->cmdSeqNum = gemm_scheduler_queue[request_queue_index].cmdSeqNum;
		for (int i = 0; i < 16; i++) {
			nvmeCmd->cmdDword[i] = gemm_scheduler_queue[request_queue_index].cmdDword[i];
		}

		//xil_printf("\r\nCMDSLOTTAG!!!! = %d\r\n",nvmeCmd->cmdSlotTag);
		recentAddress = currentAddress;
	} //MoE Rescheduling(end)
}

