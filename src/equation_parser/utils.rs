use super::{
    error::{ParserError, Result},
    symbol_counter::SymbolCounter,
    tokenizer::TokenTypes,
};

use std::{iter::Peekable, slice::Iter};

/// Apply a segment multiplier to a specific segment of the parsed formula.
///
/// # Arguments
///
/// * `stack` - The slice to which the multiplier should be applied.
/// * `mul` - The multiplier.
///
pub fn apply_multiplier(slice: &mut [SymbolCounter], mul: u32) {
    for s in slice {
        *s *= mul;
    }
}

/// Apply a segment multiplier to an entire segment of the parsed formula.
///
/// # Arguments
///
/// * `stack` - The slice to which the multiplier should be applied.
/// * `mul` - The multiplier.
///
pub fn apply_segment_multiplier(stack: &mut [Vec<SymbolCounter>], mul: &mut u32) {
    // Do we have a formula segment multiplier?
    if *mul > 0 {
        // Apply the segment multiplier to the segment.
        for segment in stack {
            apply_multiplier(segment, *mul);
        }

        *mul = 0;
    }
}

pub fn parse_number(str: &str) -> u32 {
    str.parse::<u32>().unwrap()
}

/// Sanitize an input string.
///
/// # Arguments
///
/// * `chars` - A mutable character slice.
///
pub fn sanitize(chars: &mut [char]) {
    for c in chars {
        // Subscript digits have to be normalized into their ASCII equivalents.
        let id = *c as u32;
        match id {
            // Subscript digits.
            0x2080..=0x2089 => {
                let shifted_id = id - 0x2050;
                *c = char::from_u32(shifted_id).unwrap();
            }
            // Superscript digits.
            0x2070 => *c = '0',
            0x00B9 => *c = '1',
            0x00B2 => *c = '2',
            0x00B3 => *c = '3',
            0x2074..=0x2079 => {
                // 4 - 9
                let shifted_id = id - 0x2040;
                *c = char::from_u32(shifted_id).unwrap();
            }
            // Superscript charge symbols.
            0x207A => *c = '+',
            0x207B => *c = '-',
            // Middle dot.
            0x00B7 => {
                *c = '.';
            }
            _ => {}
        }
    }
}

/// Serialize a parenthesis segment.
///
/// # Arguments
///
/// * `iter` - A mutable reference to the [`TokenTypes`] iterator.
///
pub fn serialize_parenthesis(iter: &mut Peekable<Iter<TokenTypes>>) -> Result<Vec<TokenTypes>> {
    let mut tokens = Vec::with_capacity(10);

    // The counter to indicate the depth of the parenthesis.
    let mut paren_level: usize = 1;

    // Iterate until we reach the end of the segment.
    // This one is a bit different as we need to locate the matching
    //   parenthesis. If there is a mismatch, we will panic.
    while let Some(t) = iter.peek() {
        match t {
            TokenTypes::LParen => {
                paren_level += 1;
            }
            TokenTypes::RParen => {
                paren_level -= 1;

                // We have found the matching parenthesis.
                if paren_level == 0 {
                    // We want to skip this parenthesis as it has
                    // no real value.
                    iter.next();
                    break;
                }
            }
            _ => {}
        }

        tokens.push(**t);
        iter.next();
    }

    if paren_level > 0 {
        return Err(ParserError::MismatchedParenthesis);
    }

    Ok(tokens)
}

/// Serialize a formula segment.
///
/// # Arguments
///
/// * `iter` - A mutable reference to the [`TokenTypes`] iterator.
///
pub fn serialize_segment(iter: &mut Peekable<Iter<TokenTypes>>) -> Result<Vec<TokenTypes>> {
    let mut tokens = Vec::with_capacity(10);

    // Iterate until we reach the end of the segment.
    while let Some(t) = iter.next_if(|&x| !matches!(x, TokenTypes::Dot)) {
        tokens.push(*t);
    }

    if tokens.is_empty() {
        return Err(ParserError::EmptySegment);
    }

    Ok(tokens)
}
