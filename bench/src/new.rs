use core::intrinsics::{likely, unlikely};

// the DFA array. Doesn't really need to be perfect,
#[cfg_attr(target_pointer_width = "64", repr(C, align(128)))]
#[cfg_attr(target_pointer_width = "32", repr(C, align(64)))]
struct CacheAlign<T>(T);

const ERR: u32 = 0;
const END: u32 = 17;

#[rustfmt::skip]
const DFA: &CacheAlign<[u32; 256]> = {
    const ILL: u32 = 0b00000000000000000000000000000000;
    const X00: u32 = 0b00000000001000100000000000000000;
    const XC2: u32 = 0b00000000000011000000000000000000;
    const XE0: u32 = 0b00000000000110000000000000000000;
    const XE1: u32 = 0b00000000000010000000000000000000;
    const XED: u32 = 0b00000000001011000000000000000000;
    const XF0: u32 = 0b00000000001100100000000000000000;
    const XF1: u32 = 0b00000000000100000000000000000000;
    const XF4: u32 = 0b00000000001110000000000000000000;
    const X80: u32 = 0b01000001100000000000010001100000;
    const X90: u32 = 0b00001001100000000000010001100000;
    const XA0: u32 = 0b00001000000000000110010001100000;

    &CacheAlign([
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x00 ..= 0x07
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x08 ..= 0x0A
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x10 ..= 0x17
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x18 ..= 0x1A
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x20 ..= 0x27
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x28 ..= 0x2A
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x30 ..= 0x37
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x38 ..= 0x3A
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x40 ..= 0x47
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x48 ..= 0x4A
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x50 ..= 0x57
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x58 ..= 0x5A
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x60 ..= 0x67
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x68 ..= 0x6A
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x70 ..= 0x77
        X00, X00, X00, X00, X00, X00, X00, X00, // 0x78 ..= 0x7A
        X80, X80, X80, X80, X80, X80, X80, X80, // 0x80 ..= 0x87
        X80, X80, X80, X80, X80, X80, X80, X80, // 0x88 ..= 0x8A
        X90, X90, X90, X90, X90, X90, X90, X90, // 0x90 ..= 0x97
        X90, X90, X90, X90, X90, X90, X90, X90, // 0x98 ..= 0x9A
        XA0, XA0, XA0, XA0, XA0, XA0, XA0, XA0, // 0xA0 ..= 0xA7
        XA0, XA0, XA0, XA0, XA0, XA0, XA0, XA0, // 0xA8 ..= 0xAA
        XA0, XA0, XA0, XA0, XA0, XA0, XA0, XA0, // 0xB0 ..= 0xB7
        XA0, XA0, XA0, XA0, XA0, XA0, XA0, XA0, // 0xB8 ..= 0xBA
        ILL, ILL, XC2, XC2, XC2, XC2, XC2, XC2, // 0xC0 ..= 0xC7
        XC2, XC2, XC2, XC2, XC2, XC2, XC2, XC2, // 0xC8 ..= 0xCA
        XC2, XC2, XC2, XC2, XC2, XC2, XC2, XC2, // 0xD0 ..= 0xD7
        XC2, XC2, XC2, XC2, XC2, XC2, XC2, XC2, // 0xD8 ..= 0xDA
        XE0, XE1, XE1, XE1, XE1, XE1, XE1, XE1, // 0xE0 ..= 0xE7
        XE1, XE1, XE1, XE1, XE1, XED, XE1, XE1, // 0xE8 ..= 0xEA
        XF0, XF1, XF1, XF1, XF4, ILL, ILL, ILL, // 0xF0 ..= 0xF7
        ILL, ILL, ILL, ILL, ILL, ILL, ILL, ILL, // 0xF8 ..= 0xFA
    ])
};

// Use a generic to help the compiler out some — we pass both `&[u8]` and `&[u8;
// CHUNK]` in here, and would like it to know about the constant.
#[must_use]
#[inline]
fn dfa_run(mut state: u32, chunk: impl AsRef<[u8]>) -> u32 {
    for &byte in chunk.as_ref() {
        state = DFA.0[byte as usize] >> (state & 31);
    }
    state & 31
}

// advance the DFA a single step, returning the new masked state. If you have a
// slice you should use `dfa_run` instead; it's usually more efficient.
#[must_use]
#[inline(always)]
const fn dfa_step(state: u32, byte: u8) -> u32 {
    (DFA.0[byte as usize] >> (state & 31)) & 31
}

const CHUNK: usize = 16;

#[inline]
pub const fn new_validate_utf8(inp: &[u8]) -> Result<(), Utf8Error> {
    const fn validate_utf8_const(inp: &[u8]) -> Result<(), Utf8Error> {
        simple_validate(inp, 0)
    }
    #[inline]
    fn validate_utf8_rt(inp: &[u8]) -> Result<(), Utf8Error> {
        match inp.len() {
            0 => Ok(()),
            1..=CHUNK => {
                if is_ascii_small(inp) {
                    Ok(())
                } else {
                    simple_validate(inp, 0)
                }
            }
            _ => new_validate_utf8_impl(inp),
        }
    }

    unsafe { core::intrinsics::const_eval_select((inp,), validate_utf8_const, validate_utf8_rt) }
}

// #[inline]
// fn small_nonascii_validate(inp: &[u8]) -> Result<(), Utf8Error> {
//     debug_assert!(inp.len() <= CHUNK && !inp.is_empty());
//     let mut state = END;
//     for &byte in inp {
//         state = (DFA.0[byte as usize] as u32) >> (state & 31);
//     }
//     state = state & 31;
//     // state & 31
//     // state = dfa_run(state, inp);
//     if likely(state == END) {
//         Ok(())
//     } else {
//         Err(utf8_find_error(inp, 0, false))
//         // Err(utf8_find_error(inp, inp.len() - 1, true))
//         // let res = simple_validate(inp, 0);
//         // debug_assert!(res.is_err());
//         // res
//     }
// }

/// DFA validator that can produce an accurate `Utf8Error` and which supports
/// use in `const fn`. No optimizations around ASCII or unrolling, and uses
/// slower operations to advance the DFA.
#[inline]
const fn simple_validate(input: &[u8], mut pos: usize) -> Result<(), Utf8Error> {
    let mut state = END;
    let mut valid_up_to = pos;
    while pos < input.len() {
        state = dfa_step(state, input[pos]);
        match state {
            END => valid_up_to = pos + 1,
            ERR => {
                return Err(Utf8Error {
                    valid_up_to,
                    error_len: match pos - valid_up_to {
                        // in this branch our error length must always be at least 1.
                        0 | 1 => ErrorLen::One,
                        2 => ErrorLen::Two,
                        n => {
                            debug_assert!(n == 3);
                            ErrorLen::Three
                        } // n => unreachable!("impossible errorlen: {n:?}: {pos} - {valid_up_to}"),
                    },
                });
            }
            // Keep going
            _ => {}
        }
        pos += 1;
    }
    if state != END {
        Err(Utf8Error {
            valid_up_to,
            error_len: ErrorLen::Zero,
        })
    } else {
        Ok(())
    }
}

#[inline]
fn new_validate_utf8_impl(inp: &[u8]) -> Result<(), Utf8Error> {
    let mut state = END;
    let mut pos = 0;
    let (chunks, tail) = inp.as_chunks::<CHUNK>();
    if !chunks.is_empty() {
        let mut last_state = state;
        let mut chunk_iter = chunks.iter();
        while let Some(chunk) = chunk_iter.next() {
            if state == END && all_ascii_chunk(chunk) {
                let skipped = skip_ascii_chunks(&mut chunk_iter);
                pos += skipped * CHUNK;
            } else {
                state = dfa_run(state, chunk);
                if unlikely(state == ERR) {
                    break;
                }
                last_state = state;
            }
            pos += core::mem::size_of_val(chunk);
        }
        if unlikely(state == ERR) {
            return Err(utf8_find_error(inp, pos, last_state != END));
        }
    }
    // Did we leave the optimized loop in the middle of a UTF-8 sequence?
    let was_mid_char = state != END;
    debug_assert!(state != ERR);
    if !tail.is_empty() {
        // Check and early return if the last CHUNK bytes were all ASCII. The
        // motivation here is to avoid bringing the DFA table into the cache for
        // pure ASCII.
        //
        // 1. avoid touching the DFA table for pure-ASCII input.
        // 2. add a branch into the `dfa_run` inner loop.
        //
        // So we check and see if the last CHUNK bytes were all ASCII. This does
        // compare with a few bytes that we've already processed, but handling
        // that is not required for correctness (and doing so seems ot hurt
        // performance)
        if !was_mid_char && inp.len() >= CHUNK && tail.len() < CHUNK {
            use std::convert::*;
            let range = (inp.len() - CHUNK)..inp.len();
            debug_assert!(range.contains(&pos), "{:?}", (range, pos));
            if all_ascii_chunk(inp[range].try_into().unwrap()) {
                return Ok(());
            }
        }

        state = dfa_run(state, tail);
    }

    if likely(state == END) {
        return Ok(());
    }
    let (index, backup) = if state == ERR {
        (inp.len() - tail.len(), was_mid_char)
    } else {
        (inp.len(), true)
    };
    Err(utf8_find_error(inp, index, backup))
}

#[inline]
fn backup_not_yet_invalid(inp: &[u8], mut pos: usize) -> usize {
    debug_assert!(!inp.is_empty() && inp.get(..pos).is_some());
    while pos != 0 {
        pos -= 1;
        let is_cont = (inp[pos] & 0b1100_0000) == 0b1000_0000;
        if !is_cont {
            break;
        }
    }
    pos
}

#[cold]
fn utf8_find_error(input: &[u8], mut pos: usize, backup: bool) -> Utf8Error {
    debug_assert!(!input.is_empty());
    if backup {
        pos = backup_not_yet_invalid(input, pos);
    }
    simple_validate(input, pos).unwrap_err()
}

#[inline]
fn skip_ascii_chunks(s: &mut core::slice::Iter<'_, [u8; CHUNK]>) -> usize {
    let mut i = 0;
    let initial_slice = s.as_slice();
    while let Some(c) = s.next() {
        if !all_ascii_chunk(c) {
            break;
        }
        i += 1;
    }
    *s = initial_slice[i..].iter();
    i
}

#[inline]
fn all_ascii_chunk(s: &[u8; CHUNK]) -> bool {
    // Sadly, `core::simd` currently does not compile very efficiently on some
    // targets (all the targets without simd, and some of the targets with it).
    //
    // It's also somewhat untested on others, so out of an abundance of caution
    // we avoid it on any target that isn't both:
    // - Known to support it efficiently.
    // - Actually something we'd use and test on in the versions of libcore we
    //   ship.
    const SIMD_ASCII_TEST: bool = cfg!(any(
        all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse2",
        ),
        all(target_arch = "aarch64", target_feature = "neon"),
    ));

    if SIMD_ASCII_TEST {
        use std::simd::*;
        // Workaround for <https://github.com/rust-lang/portable-simd/issues/321> :(
        let simd_chunk = Simd::<u8, CHUNK>::from_array(*s);
        if cfg!(target_arch = "aarch64") {
            simd_chunk.reduce_max() < 0x80
        } else {
            const ALL_HI: Simd<u8, CHUNK> = Simd::from_array([0x80; CHUNK]);
            const ZERO: Simd<u8, CHUNK> = Simd::from_array([0; CHUNK]);
            (simd_chunk & ALL_HI).simd_eq(ZERO).all()
        }
    } else {
        // On targets where `core::simd` doesn't compile to efficient code we
        // manually do the equivalent using u64-based SWAR using u64. Using u64 and
        // not `usize` here seems better on 32 bit which have 64 bit register
        // access, but ends up just being an extra unroll step on ones which don't
        // (so no worse, and possibly still better).
        type SwarWord = u64;
        const WORD_BYTES: usize = core::mem::size_of::<SwarWord>();
        const _: () = assert!((CHUNK % WORD_BYTES) == 0 && CHUNK != 0);
        let (arr, rem) = s.as_chunks::<WORD_BYTES>();
        debug_assert!(rem.is_empty() && !arr.is_empty());
        let mut combined = 0;
        for word_bytes in arr {
            combined |= SwarWord::from_ne_bytes(*word_bytes);
        }
        const ALL_HI: SwarWord = SwarWord::from_ne_bytes([0x80; WORD_BYTES]);
        (combined & ALL_HI) == 0
    }
}

#[inline]
fn is_ascii_small(s: &[u8]) -> bool {
    // LLVM seems to get pretty aggressive if we use a loop here, even if we
    // check the lenth first. It ends up causing performance problems (probably
    // due to lower willingness to inline. Instead of that, we handle a small
    // number of len-ranges by doing reads that intentionally overlap for some
    // of the slice lengths. Note that going overboard here will result in
    // branch prediction issues, so this is intentionally minimal -- just enough
    // to handle lengths up to `CHUNK` without pain
    match s.len() {
        0 => true,
        1..=3 => {
            // Note: If `a`, `b`, and `c` are all ASCII bytes, then `a | b | c`
            // will be too.
            let all_bytes_ored = s[0] | s[s.len() / 2] | s[s.len() - 1];
            (all_bytes_ored & 0x80) == 0
        }
        4..=16 => {
            // native endian read of a `u32`.
            // Safety: `off..(off + 4)` must be in-bounds for `n`.
            #[inline(always)]
            unsafe fn read32_unchecked(n: &[u8], off: usize) -> u32 {
                debug_assert!(n.get(off..(off + core::mem::size_of::<u32>())).is_some());
                n.as_ptr().add(off).cast::<u32>().read_unaligned()
            }
            // Safety: All these reads are guaranteed to be in-bounds for all
            // `s.len()` values in the between 4..=16 range. Sadly, the compiler
            // doesn't seem to be able to remove bounds checks on expressions
            // involving `mid_round_down` (no matter how I phrase it), so we need
            // the unsafe.
            let all_u32_ored = unsafe {
                let mid_round_down = (s.len() / 2) & !3;
                let tail = s.len() - 4;
                read32_unchecked(s, 0)
                    | read32_unchecked(s, mid_round_down)
                    | read32_unchecked(s, tail - mid_round_down)
                    | read32_unchecked(s, tail)
            };
            (all_u32_ored & 0x80808080) == 0
        }
        _ => false,
    }
}

#[derive(Copy, Eq, PartialEq, Clone, Debug)]
pub struct Utf8Error {
    pub valid_up_to: usize,
    pub error_len: ErrorLen,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ErrorLen {
    Zero = 0,
    One = 1,
    Two = 2,
    Three = 3,
}
impl From<ErrorLen> for usize {
    #[inline]
    fn from(e: ErrorLen) -> Self {
        e as usize
    }
}
impl From<ErrorLen> for Option<usize> {
    #[inline]
    fn from(e: ErrorLen) -> Self {
        if e == ErrorLen::Zero {
            None
        } else {
            Some(e as usize)
        }
    }
}

impl core::fmt::Display for Utf8Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.error_len != ErrorLen::Zero {
            write!(
                f,
                "invalid utf-8 sequence of {} bytes from index {}",
                self.error_len as u8, self.valid_up_to,
            )
        } else {
            write!(
                f,
                "incomplete utf-8 byte sequence from index {}",
                self.valid_up_to
            )
        }
    }
}

impl PartialEq<core::str::Utf8Error> for Utf8Error {
    #[inline]
    fn eq(&self, o: &core::str::Utf8Error) -> bool {
        self == &Self::from(*o)
    }
}
impl PartialEq<Utf8Error> for core::str::Utf8Error {
    #[inline]
    fn eq(&self, o: &Utf8Error) -> bool {
        o == self
    }
}

impl From<core::str::Utf8Error> for Utf8Error {
    #[inline]
    fn from(v: core::str::Utf8Error) -> Self {
        Self {
            valid_up_to: v.valid_up_to(),
            error_len: match v.error_len() {
                None => ErrorLen::Zero,
                Some(1) => ErrorLen::One,
                Some(2) => ErrorLen::Two,
                Some(3) => ErrorLen::Three,
                n => {
                    // #[cfg(debug_assertions)]
                    unreachable!("Invalid error len: {:?}", n);
                }
            },
        }
    }
}
