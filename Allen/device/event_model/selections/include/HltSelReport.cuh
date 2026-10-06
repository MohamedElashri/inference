/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "BackendCommon.h"

namespace make_selrep {
  __host__ __device__ inline void make_selrep_bank(
    unsigned* selrep,
    const unsigned* rb_objtyp,
    const unsigned* rb_hits,
    const unsigned* rb_substr,
    const unsigned* rb_stdinfo,
    const unsigned bank_size,
    const unsigned objtyp_size,
    const unsigned hits_size,
    const unsigned substr_size,
    const unsigned stdinfo_size)
  {
    const unsigned header_size = 10;
    selrep[0] = bank_size;
    // First 3 bits holds number of subbanks. For now this is fixed to 4.
    unsigned mask = 0x7L;
    unsigned bits = 3;
    unsigned n_banks = 0;
    unsigned size_iter = header_size;

    // First subbank is objtyp = 1.
    if (objtyp_size > 0) {
      n_banks++;
      selrep[1] = (selrep[1] & ~(mask << (n_banks * bits))) | (1 << (n_banks * bits));
      selrep[1 + n_banks] = size_iter + objtyp_size;
      memcpy(selrep + size_iter, rb_objtyp, objtyp_size * sizeof(unsigned));
      size_iter += objtyp_size;
    }
    // Second subbank is substr = 2.
    if (substr_size > 0) {
      n_banks++;
      selrep[1] = (selrep[1] & ~(mask << (n_banks * bits))) | (2 << (n_banks * bits));
      selrep[1 + n_banks] = size_iter + substr_size;
      memcpy(selrep + size_iter, rb_substr, substr_size * sizeof(unsigned));
      size_iter += substr_size;
    }
    // ExtraInfo subbank is substr = 3. Not filled, but minimal sub bank must be present
    // The minimal bank must house as many objects as there are in the
    // objtyp bank, which can all be empty.
    {
      n_banks++;
      selrep[1] = (selrep[1] & ~(mask << (n_banks * bits))) | (3 << (n_banks * bits));
      // Calculate the size of the empty extraInfo sub-bank from the number of objects
      const unsigned short n_objtyp = rb_objtyp[0] & 0xFFFFL;
      const unsigned n_obj = rb_objtyp[n_objtyp] & 0xFFFFL;
      const unsigned rb_einfo_size = 2 + n_obj / 4;
      selrep[1 + n_banks] = size_iter + rb_einfo_size;

      // Build the empty extraInfo bank
      // Size of the bank in the high 16 bits, number of objects in the low 16 bits
      (selrep + size_iter)[0] = (rb_einfo_size << 16) | n_obj;
      // Extra info size is stored in 8 bits pieces per object, write
      // as many empty words as needed (with padding)
      for (unsigned i_word = 1; i_word < 2 + n_obj / 4; ++i_word) {
        (selrep + size_iter)[i_word] = 0;
      }
      size_iter += rb_einfo_size;
    }
    // Third subbank is StdInfo = 4.
    if (stdinfo_size > 0) {
      n_banks++;
      selrep[1] = (selrep[1] & ~(mask << (n_banks * bits))) | (4 << (n_banks * bits));
      selrep[1 + n_banks] = size_iter + stdinfo_size;
      memcpy(selrep + size_iter, rb_stdinfo, stdinfo_size * sizeof(unsigned));
      size_iter += stdinfo_size;
    }
    // Put hits at the end because it doesn't always exist. Subbank ID = 0.
    if (hits_size > 0) {
      n_banks++;
      selrep[1] = (selrep[1] & ~(mask << (n_banks * bits))) | (0 << (n_banks * bits));
      selrep[1 + n_banks] = size_iter + hits_size;
      memcpy(selrep + size_iter, rb_hits, hits_size * sizeof(unsigned));
      size_iter += hits_size;
    }
    selrep[1] = (selrep[1] & ~mask) | n_banks;
  }
} // namespace make_selrep
