/*
 * Copyright (C) 2011 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ART_RUNTIME_VMAP_TABLE_H_
#define ART_RUNTIME_VMAP_TABLE_H_

#include "base/logging.h"
#include "leb128.h"
#include "stack.h"

namespace art {

class VmapTable {
 public:
  // For efficient encoding of special values, entries are adjusted by 2.
  static constexpr uint16_t kEntryAdjustment = 2u;
  static constexpr uint16_t kAdjustedFpMarker = static_cast<uint16_t>(0xffffu + kEntryAdjustment);
  static constexpr uint32_t kMultipleVmapsMagic = 0xFFFFFFu;

  explicit VmapTable(const uint8_t* table) : table_(table) {
  }

  VmapTable(const uint8_t* table, uint32_t dex_pc)
  : table_(FindCorrectTable(table, dex_pc)) {
  }

  VmapTable(const uint8_t* table, const StackVisitor *visitor)
  : table_(FindCorrectTable(table, visitor)) {
  }

  // Look up nth entry, not called from performance critical code.
  uint16_t operator[](size_t n) const {
    const uint8_t* table = table_;
    size_t size = DecodeUnsignedLeb128(&table);
    CHECK_LT(n, size);
    uint16_t adjusted_entry = DecodeUnsignedLeb128(&table);
    for (size_t i = 0; i < n; ++i) {
      adjusted_entry = DecodeUnsignedLeb128(&table);
    }
    return adjusted_entry - kEntryAdjustment;
  }

  size_t Size() const {
    const uint8_t* table = table_;
    return DecodeUnsignedLeb128(&table);
  }

  // Is the dex register 'vreg' in the context or on the stack? Should not be called when the
  // 'kind' is unknown or constant.
  bool IsInContext(size_t vreg, VRegKind kind, uint32_t* vmap_offset) const {
    DCHECK(kind == kReferenceVReg || kind == kIntVReg || kind == kFloatVReg ||
           kind == kLongLoVReg || kind == kLongHiVReg || kind == kDoubleLoVReg ||
           kind == kDoubleHiVReg || kind == kImpreciseConstant);
    *vmap_offset = 0xEBAD0FF5;
    // TODO: take advantage of the registers being ordered
    // TODO: we treat kImpreciseConstant as an integer below, need to ensure that such values
    //       are never promoted to floating point registers.
    bool is_float = (kind == kFloatVReg) || (kind == kDoubleLoVReg) || (kind == kDoubleHiVReg);
    bool in_floats = false;
    const uint8_t* table = table_;
    uint16_t adjusted_vreg = vreg + kEntryAdjustment;
    size_t end = DecodeUnsignedLeb128(&table);
    bool high_reg = (kind == kLongHiVReg) || (kind == kDoubleHiVReg);
    bool target64 = (kRuntimeISA == kArm64) || (kRuntimeISA == kX86_64);
    if (target64 && high_reg) {
      // Wide promoted registers are associated with the sreg of the low portion.
      adjusted_vreg--;
    }
    for (size_t i = 0; i < end; ++i) {
      // Stop if we find what we are are looking for.
      uint16_t adjusted_entry = DecodeUnsignedLeb128(&table);
      if ((adjusted_entry == adjusted_vreg) && (in_floats == is_float)) {
        *vmap_offset = i;
        return true;
      }
      // 0xffff is the marker for LR (return PC on x86), following it are spilled float registers.
      if (adjusted_entry == kAdjustedFpMarker) {
        in_floats = true;
      }
    }
    return false;
  }

  // Compute the register number that corresponds to the entry in the vmap (vmap_offset, computed
  // by IsInContext above). If the kind is floating point then the result will be a floating point
  // register number, otherwise it will be an integer register number.
  uint32_t ComputeRegister(uint32_t spill_mask, uint32_t vmap_offset, VRegKind kind) const {
    // Compute the register we need to load from the context.
    DCHECK(kind == kReferenceVReg || kind == kIntVReg || kind == kFloatVReg ||
           kind == kLongLoVReg || kind == kLongHiVReg || kind == kDoubleLoVReg ||
           kind == kDoubleHiVReg || kind == kImpreciseConstant);
    // TODO: we treat kImpreciseConstant as an integer below, need to ensure that such values
    //       are never promoted to floating point registers.
    bool is_float = (kind == kFloatVReg) || (kind == kDoubleLoVReg) || (kind == kDoubleHiVReg);
    uint32_t matches = 0;
    if (UNLIKELY(is_float)) {
      const uint8_t* table = table_;
      DecodeUnsignedLeb128(&table);  // Skip size.
      while (DecodeUnsignedLeb128(&table) != kAdjustedFpMarker) {
        matches++;
      }
      matches++;
    }
    CHECK_LT(vmap_offset - matches, static_cast<uint32_t>(POPCOUNT(spill_mask)));
    uint32_t spill_shifts = 0;
    while (matches != (vmap_offset + 1)) {
      DCHECK_NE(spill_mask, 0u);
      matches += spill_mask & 1;  // Add 1 if the low bit is set
      spill_mask >>= 1;
      spill_shifts++;
    }
    spill_shifts--;  // wind back one as we want the last match
    return spill_shifts;
  }

  bool HasMultipleVmapTables() const {
    const uint8_t* t = table_;
    size_t magic = DecodeUnsignedLeb128(&t);
    return magic == kMultipleVmapsMagic;
  }

  const uint8_t* GetMultipleVmapBase() const {
    const uint8_t* t = table_;
    size_t magic = DecodeUnsignedLeb128(&t);
    DCHECK_EQ(magic, kMultipleVmapsMagic);

    // We have multiple tables.  Find the start of the tables.
    DecodeUnsignedLeb128(&t);

    // Tables start after the size.
    return t;
  }

  const uint8_t* GetDexMap() const {
    const uint8_t* t = table_;
    size_t magic = DecodeUnsignedLeb128(&t);
    DCHECK_EQ(magic, kMultipleVmapsMagic);

    // We have multiple tables.  Find the start of the tables.
    size_t size_of_vmaps = DecodeUnsignedLeb128(&t);

    // Tables start here.
    return t + size_of_vmaps;
  }

 private:
  const uint8_t* const table_;

  const uint8_t* FindCorrectTableCommon(const uint8_t* t,
                                        const uint32_t dex_pc) {
    // We have multiple tables.  Locate the correct one for this PC.
    size_t size_of_vmaps = DecodeUnsignedLeb128(&t);
    const uint8_t* first_vmap = t;

    // Skip to start of mappings.
    t += size_of_vmaps;
    size_t num_dex_pcs = DecodeUnsignedLeb128(&t);
    for (size_t i = 0; i < num_dex_pcs; ++i) {
      // <DexPC, size of encoded table, encoded table>
      size_t dex_addr = DecodeUnsignedLeb128(&t);
      size_t offset = DecodeUnsignedLeb128(&t);
      if (dex_addr >= dex_pc) {
        return first_vmap + offset;
      }
    }
    DCHECK(false);
    return nullptr;
  }

  const uint8_t* FindCorrectTable(const uint8_t* table, const uint32_t dex_pc) {
    // Find the right Vmap table based on the Dex PC.
    // Do we have different vmap tables for each Dex PC?
    const uint8_t* t = table;
    size_t magic = DecodeUnsignedLeb128(&t);
    if (magic != kMultipleVmapsMagic) {
      return table;
    }
    return FindCorrectTableCommon(t, dex_pc);
  }

  const uint8_t* FindCorrectTable(const uint8_t* table,
                                  const StackVisitor* visitor)
                      SHARED_LOCKS_REQUIRED(Locks::mutator_lock_) {
    // Find the right Vmap table based on the Dex PC.
    // Do we have different vmap tables for each Dex PC?
    const uint8_t* t = table;
    size_t magic = DecodeUnsignedLeb128(&t);
    if (magic != kMultipleVmapsMagic) {
      return table;
    }
    return FindCorrectTableCommon(t, visitor->GetDexPc());
  }
};

}  // namespace art

#endif  // ART_RUNTIME_VMAP_TABLE_H_
