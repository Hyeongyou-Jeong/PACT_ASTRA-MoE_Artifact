//////////////////////////////////////////////////////////////////////////////////
// io_access.h for Cosmos+ OpenSSD
// Copyright (c) 2016 Hanyang University ENC Lab.
// Contributed by Yong Ho Song <yhsong@enc.hanyang.ac.kr>
//				  Youngjin Jo <yjjo@enc.hanyang.ac.kr>
//				  Sangjin Lee <sjlee@enc.hanyang.ac.kr>
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

//////////////////////////////////////////////////////////////////////////////////
// Company: ENC Lab. <http://enc.hanyang.ac.kr>
// Engineer: Sangjin Lee <sjlee@enc.hanyang.ac.kr>
//
// Project Name: Cosmos+ OpenSSD
// Design Name: Cosmos+ Firmware
// Module Name: IO Access Mate
// File Name: io_access.h
//
// Version: v1.0.0
//
// Description:
//   - defines IO read/write macros
//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
// Revision History:
//
// * v1.0.0
//   - First draft
//////////////////////////////////////////////////////////////////////////////////


#ifndef __IO_ACCESS_H_
#define __IO_ACCESS_H_

#define IO_WRITE32(addr, val)		*((volatile unsigned int *)(addr)) = val
#define IO_READ32(addr)				*((volatile unsigned int *)(addr))

#define IO_WRITE16(addr, val)   (*((volatile unsigned short *)(addr)) = (val))
#define IO_READ16(addr)         (*((volatile unsigned short *)(addr)))

#define IO_WRITE8(addr, val)   (*((volatile unsigned char *)(addr)) = (val))
#define IO_READ8(addr)         (*((volatile unsigned char *)(addr)))

#define PACK_TWO_4BIT(high, low) ((((high) & 0x0F) << 4) | ((low) & 0x0F))

#define PACK_FOUR_4BIT(b1, b2, b3, b4) ((((b1) & 0x0F) << 12) | (((b2) & 0x0F) << 8) | (((b3) & 0x0F) << 4) | ((b4) & 0x0F))


#define GET_HIGH_NIBBLE(byte) (((byte) >> 4) & 0x0F)
#define GET_LOW_NIBBLE(byte) ((byte) & 0x0F)

#endif	//__IO_ACCESS_H_
