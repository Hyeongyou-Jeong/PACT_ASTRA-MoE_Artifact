/*
 * gemm_scheduler.h
 *
 *  Created on: 2024. 12. 17.
 *      Author: hgjeong
 */

#ifndef __GEMM_SCHEDULER_H_
#define __GEMM_SCHEDULER_H_

#define SCHEDULER_WINDOW 16
#define SCHEDULING_ALGORITHM 1//0: FIFO, 1: MoE Rescheduling

void reset_gemm_scheduler_queue();
void enqueue_gemm_scheduler();
void fetch_request_from_scheduler_queue();


typedef struct _SCHEDULER_COMMAND
{
	unsigned short cmdValid;
	int qOrder;
	unsigned short qID;
	unsigned short cmdSlotTag;
	unsigned int cmdSeqNum;
	unsigned int cmdDword[16];
}SCHEDULER_COMMAND;

#endif
