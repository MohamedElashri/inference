/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <AllenBuffer.cuh>
#include <Datatype.cuh>

using mask_vec_t = Allen::device_buffer<mask_t>;

class EventMask {
public:
  using Word = uint64_t;
  static constexpr size_t BITS_PER_WORD = sizeof(Word) * 8;

  class ActiveEventIterator; // Forward declaration of iterator

  explicit EventMask(size_t max_events) :
    m_max_events(max_events), m_words((max_events + BITS_PER_WORD - 1) / BITS_PER_WORD, 0)
  {}

  unsigned max_events() const { return m_max_events; }

  // Bit manipulation
  void set(size_t bit) { m_words[bit / BITS_PER_WORD] |= (Word(1) << (bit % BITS_PER_WORD)); }

  void clear(size_t bit) { m_words[bit / BITS_PER_WORD] &= ~(Word(1) << (bit % BITS_PER_WORD)); }

  bool test(size_t bit) const { return (m_words[bit / BITS_PER_WORD] >> (bit % BITS_PER_WORD)) & 1; }

  void reset() { std::fill(m_words.begin(), m_words.end(), 0); }

  void fill()
  {
    std::fill(m_words.begin(), m_words.end(), ~Word(0));
    if (size_t valid_bits = m_max_events % BITS_PER_WORD) {
      m_words.back() &= ((Word(1) << valid_bits) - 1);
    }
  }

  // Set operations
  EventMask operator&(const EventMask& other) const
  {
    EventMask result(m_max_events);
    for (size_t i = 0; i < m_words.size(); i++) {
      result.m_words[i] = m_words[i] & other.m_words[i];
    }
    return result;
  }

  EventMask operator|(const EventMask& other) const
  {
    EventMask result(m_max_events);
    for (size_t i = 0; i < m_words.size(); i++) {
      result.m_words[i] = m_words[i] | other.m_words[i];
    }
    return result;
  }

  EventMask operator~() const
  {
    EventMask result(m_max_events);
    for (size_t i = 0; i < m_words.size(); i++) {
      result.m_words[i] = ~m_words[i];
    }
    if (size_t valid_bits = m_max_events % BITS_PER_WORD) {
      result.m_words.back() &= ((Word(1) << valid_bits) - 1);
    }
    return result;
  }

  EventMask& operator&=(const EventMask& other)
  {
    for (size_t i = 0; i < m_words.size(); i++) {
      m_words[i] &= other.m_words[i];
    }
    return *this;
  }

  EventMask& operator|=(const EventMask& other)
  {
    for (size_t i = 0; i < m_words.size(); i++) {
      m_words[i] |= other.m_words[i];
    }
    return *this;
  }

  size_t popcount() const
  {
    size_t count = 0;
    for (Word w : m_words)
      count += std::popcount(w);
    return count;
  }

  bool any() const
  {
    for (Word w : m_words) {
      if (w) return true;
    }
    return false;
  }

  bool none() const { return !any(); }

  // Event list conversion
  void to_event_list(Allen::host_buffer<mask_t>& indices) const
  {
    indices.resize(popcount());
    size_t idx = 0;
    for (size_t i = 0; i < m_words.size(); i++) {
      Word w = m_words[i];
      unsigned base = i * BITS_PER_WORD;
      while (w) {
        indices[idx++] = mask_t {base + std::countr_zero(w)};
        w &= w - 1; // clear lowest bit
      }
    }
  }

  void from_event_list(const Allen::host_buffer<mask_t>& indices)
  {
    reset();
    for (const mask_t i : indices) {
      set(i);
    }
  }

  // Debug print
  friend std::ostream& operator<<(std::ostream& os, const EventMask& m)
  {
    for (size_t i = 0; i < m.m_max_events; ++i) {
      os << (m.test(i) ? '1' : '0');
    }
    os << " (" << m.popcount() << " / " << m.m_max_events << ")";
    return os;
  }

  // Core iteration methods
  size_t find_first() const
  {
    for (size_t i = 0; i < m_words.size(); i++) {
      if (m_words[i] != 0) {
        return i * BITS_PER_WORD + std::countr_zero(m_words[i]);
      }
    }
    return m_max_events;
  }

  size_t find_next(size_t current) const
  {
    if (current >= m_max_events - 1) return m_max_events;
    // Start searching from the next bit
    size_t next_bit = current + 1;
    size_t word_idx = next_bit / BITS_PER_WORD;
    size_t bit_in_word = next_bit % BITS_PER_WORD;
    // Check current word first (masking out bits before next_bit)
    if (word_idx < m_words.size()) {
      Word word = m_words[word_idx];
      if (bit_in_word > 0) {
        word &= ~((Word(1) << bit_in_word) - 1);
      }
      if (word != 0) {
        return word_idx * BITS_PER_WORD + std::countr_zero(word);
      }
      ++word_idx;
    }
    // Check remaining words
    for (size_t i = word_idx; i < m_words.size(); i++) {
      if (m_words[i] != 0) {
        return i * BITS_PER_WORD + std::countr_zero(m_words[i]);
      }
    }
    return m_max_events;
  }

  // Iterator
  ActiveEventIterator begin() const { return ActiveEventIterator(*this, find_first()); }

  ActiveEventIterator end() const { return ActiveEventIterator(*this, m_max_events); }

  class ActiveEventIterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = const size_t*;
    using reference = size_t;

    ActiveEventIterator(const EventMask& mask, size_t current_bit) : m_mask(&mask), m_current_bit(current_bit) {}

    ActiveEventIterator& operator++()
    {
      if (m_current_bit < m_mask->max_events()) {
        m_current_bit = m_mask->find_next(m_current_bit);
      }
      return *this;
    }

    size_t operator*() const { return m_current_bit; }

    bool operator==(const ActiveEventIterator& other) const
    {
      return m_mask == other.m_mask && m_current_bit == other.m_current_bit;
    }

    bool operator!=(const ActiveEventIterator& other) const { return !(*this == other); }

  private:
    const EventMask* m_mask;
    size_t m_current_bit;
  };

private:
  unsigned m_max_events;
  std::vector<Word> m_words;
};
