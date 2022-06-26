use super::{
    parser_error::{ParserError, Result},
    symbol_counter::SymbolCounter,
    tokenizer::TokenTypes,
};

use std::{iter::Peekable, slice::Iter};

/// Apply a coefficient to a specific segment of the parsed formula.
///
/// # Arguments
///
/// * `stack` - The slice to which the multiplier should be applied.
/// * `coeff` - The coefficient.
///
pub fn apply_coefficient(slice: &mut [SymbolCounter], coeff: u32) {
    for s in slice {
        *s *= coeff;
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
            // Subscript charge symbols.
            0x208A => *c = '+',
            0x208B => *c = '-',
            _ => {}
        }
    }
}

fn matching_closing_parenthesis(c: char) -> char {
    match c {
        '(' => ')',
        '[' => ']',
        '{' => '}',
        _ => c,
    }
}

/// Serialize a parenthesis segment.
///
/// # Arguments
///
/// * `iter` - A mutable reference to the [`TokenTypes`] iterator.
///
pub fn serialize_parenthesis(
    iter: &mut Peekable<Iter<TokenTypes>>,
    paren_type: char,
) -> Result<Vec<TokenTypes>> {
    let mut tokens = Vec::with_capacity(10);

    // The counter to indicate the depth of the parenthesis.
    let mut paren_level: usize = 1;

    // Iterate until we reach the end of the segment.
    // This one is a bit different as we need to locate the matching
    //   parenthesis. If there is a mismatch, we will panic.
    while let Some(t) = iter.peek() {
        match t {
            TokenTypes::LParen(c) => {
                // Is this parenthesis of a matching type?
                if paren_type == *c {
                    paren_level += 1;
                }
            }
            TokenTypes::RParen(c) => {
                // Is this parenthesis of a matching type?
                if *c == matching_closing_parenthesis(paren_type) {
                    paren_level -= 1;
                }

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
