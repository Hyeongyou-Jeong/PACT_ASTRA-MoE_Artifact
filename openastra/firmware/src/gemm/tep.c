/*
 * tep.c - Tile-based Expert Processing scheduler
 *
 * Async flash + NPU double-buffer over the native fixed-shape GEMM tile.
 */

#include "xil_printf.h"
#include "stdio.h"
#include "string.h"

#include "tep.h"
#include "gemm_dispatcher.h"
#include "npu_driver.h"

#include "../nvme/host_lld.h"
#include "../nvme/nvme.h"
#include "../memory_map.h"
#include "../request_format.h"
#include "../request_allocation.h"
#include "../request_schedule.h"
#include "../request_transform.h"
#include "../address_translation.h"
#include "../ftl_config.h"

#if TEP_DEBUG
#define TEP_LOG(...)	xil_printf(__VA_ARGS__)
#else
#define TEP_LOG(...)	do { } while (0)
#endif

static TEP_REQUEST g_tepReq;
static TEP_SLOT g_tepSlot[TEP_WEIGHT_SLOT_COUNT];
static TEP_INFLIGHT_PAGE g_tepInflight[TEP_MAX_INFLIGHT_PAGES];
static unsigned int g_tepNpuBusy;

static void TepYieldFlashProgress(void);
static void TepIssuePendingTileReads(void);
static void TepDispatchReadyTile(void);
static void TepTryCompleteRequest(void);
static void TepTryFinishAbort(void);
static void TepAbortRequest(unsigned short statusFieldWord);
static int TepAllocSlot(void);
static void TepFreeSlot(unsigned int slot);
static void TepIssueTileFlash(unsigned int tileIndex, unsigned int slot);
static void TepRunNativeNpu(unsigned int weightAddr, unsigned int ifmapAddr, unsigned int ofmapAddr);
static int TepRegisterInflight(unsigned int reqSlotTag, unsigned int tileIndex, unsigned int pageIndex);
static void TepReleaseTileAfterDrain(unsigned int tileIndex);
static void TepResetRequestIdle(void);

void InitTep(void)
{
	unsigned int i;

	memset(&g_tepReq, 0, sizeof(g_tepReq));
	g_tepReq.state = TEP_REQ_IDLE;
	g_tepReq.cmdSlotTag = TEP_REQ_NONE;

	for (i = 0; i < TEP_WEIGHT_SLOT_COUNT; i++) {
		g_tepSlot[i].state = TEP_SLOT_FREE;
		g_tepSlot[i].tileIndex = TEP_TILE_NONE;
		g_tepSlot[i].dramAddr = WEIGHT_BUFFER_BASE_ADDR + i * TEP_WEIGHT_SLOT_BYTES;
	}

	for (i = 0; i < TEP_MAX_INFLIGHT_PAGES; i++) {
		g_tepInflight[i].valid = 0;
		g_tepInflight[i].reqSlotTag = REQ_SLOT_TAG_NONE;
	}

	g_tepNpuBusy = 0;
	npu_poll_yield = TepYieldFlashProgress;

#if TEP_SOFTWARE_TEST
	TepRunSoftwareTests();
#endif
}

int TepIsActive(void)
{
	return (g_tepReq.state == TEP_REQ_ACTIVE) ||
		(g_tepReq.state == TEP_REQ_ABORTING) ||
		(g_tepReq.state == TEP_REQ_COMPLETING);
}

static void TepYieldFlashProgress(void)
{
	CheckDoneNvmeDmaReq();
	SchedulingNandReq();
}

static int TepAllocSlot(void)
{
	unsigned int i;
	for (i = 0; i < TEP_WEIGHT_SLOT_COUNT; i++) {
		if (g_tepSlot[i].state == TEP_SLOT_FREE)
			return (int)i;
	}
	return -1;
}

static void TepFreeSlot(unsigned int slot)
{
	if (slot >= TEP_WEIGHT_SLOT_COUNT)
		return;
	g_tepSlot[slot].state = TEP_SLOT_FREE;
	g_tepSlot[slot].tileIndex = TEP_TILE_NONE;
}

static int TepRegisterInflight(unsigned int reqSlotTag, unsigned int tileIndex, unsigned int pageIndex)
{
	unsigned int i;
	for (i = 0; i < TEP_MAX_INFLIGHT_PAGES; i++) {
		if (!g_tepInflight[i].valid) {
			g_tepInflight[i].valid = 1;
			g_tepInflight[i].reqSlotTag = (unsigned short)reqSlotTag;
			g_tepInflight[i].tileIndex = (unsigned short)tileIndex;
			g_tepInflight[i].pageIndex = (unsigned short)pageIndex;
			return 0;
		}
	}
	return -1;
}

static void TepResetRequestIdle(void)
{
	unsigned int i;

	g_tepReq.state = TEP_REQ_IDLE;
	g_tepReq.cmdSlotTag = TEP_REQ_NONE;
	g_tepReq.tileCount = 0;
	g_tepReq.completedTileCount = 0;
	g_tepReq.nextIssueTile = 0;
	g_tepReq.outstandingPageCount = 0;
	g_tepReq.abortStatusWord = 0;
	g_tepReq.cplSent = 0;
	for (i = 0; i < TEP_MAX_TILES; i++) {
		g_tepReq.tile[i].state = TEP_TILE_FREE;
		g_tepReq.tile[i].bufferSlot = TEP_SLOT_NONE;
	}
	for (i = 0; i < TEP_WEIGHT_SLOT_COUNT; i++) {
		g_tepSlot[i].state = TEP_SLOT_FREE;
		g_tepSlot[i].tileIndex = TEP_TILE_NONE;
	}
	for (i = 0; i < TEP_MAX_INFLIGHT_PAGES; i++) {
		g_tepInflight[i].valid = 0;
		g_tepInflight[i].reqSlotTag = REQ_SLOT_TAG_NONE;
	}
	g_tepNpuBusy = 0;
}

static void TepAbortRequest(unsigned short statusFieldWord)
{
	if (g_tepReq.state != TEP_REQ_ACTIVE && g_tepReq.state != TEP_REQ_ABORTING)
		return;

	if (g_tepReq.state == TEP_REQ_ACTIVE) {
		g_tepReq.state = TEP_REQ_ABORTING;
		g_tepReq.abortStatusWord = statusFieldWord;
		TEP_LOG("[TEP] req=%u ABORTING status=0x%x outstanding=%u\r\n",
			g_tepReq.cmdSlotTag, statusFieldWord, g_tepReq.outstandingPageCount);
	}

	TepTryFinishAbort();
}

static void TepTryFinishAbort(void)
{
	unsigned int i;

	if (g_tepReq.state != TEP_REQ_ABORTING)
		return;
	if (g_tepReq.outstandingPageCount != 0)
		return;
	if (g_tepNpuBusy)
		return;

	for (i = 0; i < g_tepReq.tileCount; i++) {
		unsigned int slot = g_tepReq.tile[i].bufferSlot;
		if (slot < TEP_WEIGHT_SLOT_COUNT)
			TepFreeSlot(slot);
		g_tepReq.tile[i].bufferSlot = TEP_SLOT_NONE;
		g_tepReq.tile[i].state = TEP_TILE_FREE;
	}

	if (!g_tepReq.cplSent) {
		g_tepReq.state = TEP_REQ_COMPLETING;
		set_auto_nvme_cpl(g_tepReq.cmdSlotTag, 0, g_tepReq.abortStatusWord);
		g_tepReq.cplSent = 1;
		TEP_LOG("[TEP] req=%u error-complete\r\n", g_tepReq.cmdSlotTag);
	}

	TepResetRequestIdle();
}

static void TepReleaseTileAfterDrain(unsigned int tileIndex)
{
	TEP_TILE *tile;

	if (tileIndex >= g_tepReq.tileCount)
		return;
	tile = &g_tepReq.tile[tileIndex];

	/* Keep slot until all pages of this tile have drained. */
	if (tile->completedPages < tile->expectedPages)
		return;

	if (tile->bufferSlot < TEP_WEIGHT_SLOT_COUNT)
		TepFreeSlot(tile->bufferSlot);
	tile->bufferSlot = TEP_SLOT_NONE;
	tile->state = TEP_TILE_FREE;
}

int TepStartRequest(unsigned int cmdSlotTag,
		unsigned int ifmapAddr,
		unsigned int ofmapAddr,
		unsigned int expertBaseLba,
		unsigned int numRows,
		unsigned int outCols)
{
	unsigned int i;
	unsigned int tileCount;
	unsigned int baseSlice;
	unsigned int outputBytes;
	unsigned int ofmapEnd = WEIGHT_BUFFER_BASE_ADDR;
	unsigned int ifmapEnd = OFMAP_BUFFER_BASE_ADDR;

	if (TepIsActive())
		return TEP_ERR_BUSY;

	if (numRows == 0)
		numRows = NPU_TILE_M;
	if (outCols == 0)
		outCols = NPU_TILE_N;

	if (numRows != NPU_TILE_M)
		return TEP_ERR_ROWS;
	if ((outCols % NPU_TILE_N) != 0)
		return TEP_ERR_COLS;

	tileCount = outCols / NPU_TILE_N;
	if (tileCount == 0 || tileCount > TEP_MAX_TILES)
		return TEP_ERR_TILECNT;

	/* Expert base must be slice-aligned: baseSlice = LBA / NVME_BLOCKS_PER_SLICE. */
	if ((expertBaseLba % NVME_BLOCKS_PER_SLICE) != 0)
		return TEP_ERR_LBA_ALIGN;

	outputBytes = tileCount * NPU_TILE_OFMAP_BYTES;

	if (ofmapAddr < OFMAP_BUFFER_BASE_ADDR || ofmapAddr > ofmapEnd)
		return TEP_ERR_OFMAP;
	if (outputBytes > (ofmapEnd - ofmapAddr))
		return TEP_ERR_OFMAP;

	if (ifmapAddr < IFMAP_BUFFER_BASE_ADDR || ifmapAddr > ifmapEnd)
		return TEP_ERR_IFMAP;
	if (NPU_TILE_IFMAP_BYTES > (ifmapEnd - ifmapAddr))
		return TEP_ERR_IFMAP;

	if ((ofmapAddr & 0x1) != 0)
		return TEP_ERR_OFMAP;
	if ((ifmapAddr & 0x1) != 0)
		return TEP_ERR_IFMAP;

	baseSlice = expertBaseLba / NVME_BLOCKS_PER_SLICE;

	memset(&g_tepReq, 0, sizeof(g_tepReq));
	g_tepReq.state = TEP_REQ_ACTIVE;
	g_tepReq.cmdSlotTag = (unsigned short)cmdSlotTag;
	g_tepReq.tileCount = (unsigned short)tileCount;
	g_tepReq.completedTileCount = 0;
	g_tepReq.nextIssueTile = 0;
	g_tepReq.cplSent = 0;
	g_tepReq.outstandingPageCount = 0;
	g_tepReq.abortStatusWord = 0;
	g_tepReq.expertBaseLba = expertBaseLba;
	g_tepReq.ifmapBaseAddr = ifmapAddr;
	g_tepReq.ofmapBaseAddr = ofmapAddr;
	g_tepReq.numRows = numRows;
	g_tepReq.outCols = outCols;

	for (i = 0; i < tileCount; i++) {
		g_tepReq.tile[i].tileIndex = (unsigned short)i;
		g_tepReq.tile[i].state = TEP_TILE_ALLOCATED;
		g_tepReq.tile[i].bufferSlot = TEP_SLOT_NONE;
		g_tepReq.tile[i].expectedPages = FLASH_PAGES_PER_TILE;
		g_tepReq.tile[i].completedPages = 0;
		g_tepReq.tile[i].pageSeen = 0;
		g_tepReq.tile[i].weightSliceBase = baseSlice + i * FLASH_PAGES_PER_TILE;
		g_tepReq.tile[i].ifmapAddr = ifmapAddr;
		g_tepReq.tile[i].ofmapAddr = ofmapAddr + i * NPU_TILE_OFMAP_BYTES;
	}
	for (; i < TEP_MAX_TILES; i++) {
		g_tepReq.tile[i].state = TEP_TILE_FREE;
		g_tepReq.tile[i].bufferSlot = TEP_SLOT_NONE;
	}

	for (i = 0; i < TEP_WEIGHT_SLOT_COUNT; i++) {
		g_tepSlot[i].state = TEP_SLOT_FREE;
		g_tepSlot[i].tileIndex = TEP_TILE_NONE;
	}

	TEP_LOG("\r\n[TEP] req=%u start tiles=%u M=%u N=%u baseLba=0x%x\r\n",
		cmdSlotTag, tileCount, numRows, outCols, expertBaseLba);

	TepIssuePendingTileReads();
	return TEP_OK;
}

static void TepIssueTileFlash(unsigned int tileIndex, unsigned int slot)
{
	unsigned int page;
	TEP_TILE *tile = &g_tepReq.tile[tileIndex];
	NVME_COMPLETION cpl;

	if (g_tepReq.state != TEP_REQ_ACTIVE)
		return;

	tile->bufferSlot = (unsigned short)slot;
	tile->completedPages = 0;
	tile->pageSeen = 0;
	tile->state = TEP_TILE_FLASH_ISSUED;

	g_tepSlot[slot].state = TEP_SLOT_FILLING;
	g_tepSlot[slot].tileIndex = (unsigned short)tileIndex;

	TEP_LOG("[TEP] req=%u tile=%u issue pages=%u slot=%u\r\n",
		g_tepReq.cmdSlotTag, tileIndex, tile->expectedPages, slot);

	cpl.statusFieldWord = 0;
	cpl.statusField.SC = SC_INVALID_FIELD_IN_COMMAND;
	cpl.statusField.SCT = SCT_GENERIC_COMMAND_STATUS;

	for (page = 0; page < tile->expectedPages; page++) {
		unsigned int sliceAddr = tile->weightSliceBase + page;
		unsigned int vsa;
		unsigned int reqSlotTag;
		unsigned int bufAddr;

		if (g_tepReq.state != TEP_REQ_ACTIVE)
			return;

		vsa = logicalSliceMapPtr->logicalSlice[sliceAddr].virtualSliceAddr;
		if (vsa == VSA_NONE) {
			xil_printf("[TEP] ERROR unmapped slice 0x%x tile=%u\r\n", sliceAddr, tileIndex);
			/* Keep slot until any already-issued pages of this/other tiles drain. */
			TepAbortRequest(cpl.statusFieldWord);
			return;
		}

		bufAddr = g_tepSlot[slot].dramAddr + page * BYTES_PER_DATA_REGION_OF_SLICE;
		reqSlotTag = GetFromFreeReqQ();

		reqPoolPtr->reqPool[reqSlotTag].reqType = REQ_TYPE_NAND;
		reqPoolPtr->reqPool[reqSlotTag].reqCode = REQ_CODE_READ;
		reqPoolPtr->reqPool[reqSlotTag].nvmeCmdSlotTag = g_tepReq.cmdSlotTag;
		reqPoolPtr->reqPool[reqSlotTag].logicalSliceAddr = sliceAddr;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.dataBufFormat = REQ_OPT_DATA_BUF_ADDR;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.nandAddr = REQ_OPT_NAND_ADDR_VSA;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.nandEcc = REQ_OPT_NAND_ECC_ON;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.nandEccWarning = REQ_OPT_NAND_ECC_WARNING_OFF;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.rowAddrDependencyCheck = REQ_OPT_ROW_ADDR_DEPENDENCY_CHECK;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.blockSpace = REQ_OPT_BLOCK_SPACE_MAIN;
		reqPoolPtr->reqPool[reqSlotTag].reqOpt.gemmCmd = 1;
		reqPoolPtr->reqPool[reqSlotTag].dataBufInfo.addr = bufAddr;
		reqPoolPtr->reqPool[reqSlotTag].nandInfo.virtualSliceAddr = vsa;

		/* Register BEFORE enqueue �� never issue untracked TEP NAND. */
		if (TepRegisterInflight(reqSlotTag, tileIndex, page) != 0) {
			xil_printf("[TEP] ERROR inflight table full\r\n");
			reqPoolPtr->reqPool[reqSlotTag].reqOpt.gemmCmd = 0;
			PutToFreeReqQ(reqSlotTag);
			TepAbortRequest(cpl.statusFieldWord);
			return;
		}

		g_tepReq.outstandingPageCount++;
		SelectLowLevelReqQ(reqSlotTag);
		PushToGemmNandRequestTable(reqSlotTag);
	}
}

static void TepIssuePendingTileReads(void)
{
	if (g_tepReq.state != TEP_REQ_ACTIVE)
		return;

	while (g_tepReq.nextIssueTile < g_tepReq.tileCount) {
		int slot = TepAllocSlot();
		if (slot < 0)
			break;
		TepIssueTileFlash(g_tepReq.nextIssueTile, (unsigned int)slot);
		g_tepReq.nextIssueTile++;
		if (g_tepReq.state != TEP_REQ_ACTIVE)
			break;
	}
}

void TepNotifyPageCompletion(unsigned int reqSlotTag)
{
	unsigned int i;
	unsigned int tileIndex;
	unsigned int pageIndex;
	TEP_TILE *tile;
	unsigned short found = 0;

	tileIndex = TEP_TILE_NONE;
	pageIndex = 0;
	for (i = 0; i < TEP_MAX_INFLIGHT_PAGES; i++) {
		if (g_tepInflight[i].valid && g_tepInflight[i].reqSlotTag == reqSlotTag) {
			tileIndex = g_tepInflight[i].tileIndex;
			pageIndex = g_tepInflight[i].pageIndex;
			g_tepInflight[i].valid = 0;
			g_tepInflight[i].reqSlotTag = REQ_SLOT_TAG_NONE;
			found = 1;
			break;
		}
	}

	if (!found)
		return;

	if (g_tepReq.outstandingPageCount > 0)
		g_tepReq.outstandingPageCount--;

	if (tileIndex >= g_tepReq.tileCount) {
		TepTryFinishAbort();
		return;
	}

	tile = &g_tepReq.tile[tileIndex];

	if (g_tepReq.state == TEP_REQ_ABORTING) {
		if (pageIndex < 16) {
			unsigned short bit = (unsigned short)(1u << pageIndex);
			if ((tile->pageSeen & bit) == 0) {
				tile->pageSeen |= bit;
				if (tile->completedPages < tile->expectedPages)
					tile->completedPages++;
			}
		} else if (tile->completedPages < tile->expectedPages) {
			tile->completedPages++;
		}
		TepReleaseTileAfterDrain(tileIndex);
		TepTryFinishAbort();
		return;
	}

	if (g_tepReq.state != TEP_REQ_ACTIVE) {
#if TEP_DEBUG
		xil_printf("[TEP] unexpected notify state=%u tag=%u\r\n", g_tepReq.state, reqSlotTag);
#endif
		return;
	}

	if (tile->state != TEP_TILE_FLASH_ISSUED)
		return;

	if (pageIndex < 16) {
		unsigned short bit = (unsigned short)(1u << pageIndex);
		if (tile->pageSeen & bit)
			return;
		tile->pageSeen |= bit;
	}

	tile->completedPages++;
	if (tile->completedPages < tile->expectedPages)
		return;

	tile->state = TEP_TILE_FLASH_READY;
	if (tile->bufferSlot < TEP_WEIGHT_SLOT_COUNT)
		g_tepSlot[tile->bufferSlot].state = TEP_SLOT_READY;

	TEP_LOG("[TEP] req=%u tile=%u flash-ready\r\n", g_tepReq.cmdSlotTag, tileIndex);
}

static void TepRunNativeNpu(unsigned int weightAddr, unsigned int ifmapAddr, unsigned int ofmapAddr)
{
	fetch_npu_request_quiet(weightAddr, ifmapAddr, ofmapAddr);
}

static void TepDispatchReadyTile(void)
{
	unsigned int i;
	unsigned int best = TEP_TILE_NONE;
	TEP_TILE *tile;
	unsigned int slot;

	if (g_tepReq.state != TEP_REQ_ACTIVE || g_tepNpuBusy)
		return;

	for (i = 0; i < g_tepReq.tileCount; i++) {
		if (g_tepReq.tile[i].state == TEP_TILE_FLASH_READY) {
			best = i;
			break;
		}
	}
	if (best == TEP_TILE_NONE)
		return;

	tile = &g_tepReq.tile[best];
	slot = tile->bufferSlot;
	if (slot >= TEP_WEIGHT_SLOT_COUNT)
		return;

	TepIssuePendingTileReads();
	if (g_tepReq.state != TEP_REQ_ACTIVE)
		return;
	if (tile->state != TEP_TILE_FLASH_READY)
		return;

	tile->state = TEP_TILE_NPU_RUNNING;
	g_tepSlot[slot].state = TEP_SLOT_NPU_BUSY;
	g_tepNpuBusy = 1;

	TEP_LOG("[TEP] req=%u tile=%u npu-start\r\n", g_tepReq.cmdSlotTag, best);

	TepIssuePendingTileReads();

	TepRunNativeNpu(g_tepSlot[slot].dramAddr, tile->ifmapAddr, tile->ofmapAddr);

	tile->state = TEP_TILE_DONE;
	g_tepReq.completedTileCount++;
	TepFreeSlot(slot);
	g_tepNpuBusy = 0;

	TEP_LOG("[TEP] req=%u tile=%u done (%u/%u)\r\n",
		g_tepReq.cmdSlotTag, best, g_tepReq.completedTileCount, g_tepReq.tileCount);

	if (g_tepReq.state == TEP_REQ_ABORTING) {
		TepTryFinishAbort();
		return;
	}

	TepIssuePendingTileReads();
	TepTryCompleteRequest();
}

static void TepTryCompleteRequest(void)
{
	if (g_tepReq.state != TEP_REQ_ACTIVE)
		return;
	if (g_tepReq.completedTileCount != g_tepReq.tileCount)
		return;
	if (g_tepReq.outstandingPageCount != 0)
		return;
	if (g_tepReq.cplSent)
		return;

	g_tepReq.state = TEP_REQ_COMPLETING;
	set_auto_nvme_cpl(g_tepReq.cmdSlotTag, 0, 0);
	g_tepReq.cplSent = 1;

	TEP_LOG("[TEP] req=%u complete\r\n", g_tepReq.cmdSlotTag);

	TepResetRequestIdle();
}

void TepService(void)
{
	unsigned int i;
	unsigned int before;

	if (!TepIsActive())
		return;

	CheckDoneNvmeDmaReq();
	SchedulingNandReq();

	if (g_tepReq.state == TEP_REQ_ABORTING) {
		TepTryFinishAbort();
		return;
	}

	TepIssuePendingTileReads();

	for (i = 0; i < TEP_MAX_TILES && g_tepReq.state == TEP_REQ_ACTIVE; i++) {
		before = g_tepReq.completedTileCount;
		TepDispatchReadyTile();
		if (g_tepReq.completedTileCount == before)
			break;
		TepIssuePendingTileReads();
	}

	TepTryCompleteRequest();
	TepTryFinishAbort();
}
