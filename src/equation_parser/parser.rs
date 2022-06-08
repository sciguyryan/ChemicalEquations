use crate::definitions::enums::Symbol;

use std::{collections::HashMap, iter::Peekable, ops::MulAssign, slice::Iter, str::FromStr};

use super::{
    error::{ParserError, Result},
    tokenizer::{self, TokenTypes},
};

#[derive(Debug, Clone)]
struct SymbolCounter {
    pub symbol: Symbol,
    pub count: u32,
}

impl SymbolCounter {
    pub fn new(symbol: Symbol, count: u32) -> Self {
        Self { symbol, count }
    }
}

impl MulAssign<u32> for SymbolCounter {
    fn mul_assign(&mut self, rhs: u32) {
        self.count *= rhs;
    }
}

/// Apply a segment multiplier to a specific segment of the parsed formula.
///
/// # Arguments
///
/// * `stack` - The slice to which the multiplier should be applied.
/// * `mul` - The multiplier.
///
fn apply_multiplier(slice: &mut [SymbolCounter], mul: u32) {
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
fn apply_segment_multiplier(stack: &mut [Vec<SymbolCounter>], mul: &mut u32) {
    // Do we have a formula segment multiplier?
    if *mul > 0 {
        // Apply the segment multiplier to the segment.
        for segment in stack {
            apply_multiplier(segment, *mul);
        }

        *mul = 0;
    }
}

/// Serialize a parenthesis segment.
///
/// # Arguments
///
/// * `iter` - A mutable reference to the [`TokenTypes`] iterator.
///
fn serialize_parenthesis(iter: &mut Peekable<Iter<TokenTypes>>) -> Result<Vec<TokenTypes>> {
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
fn serialize_segment(iter: &mut Peekable<Iter<TokenTypes>>) -> Result<Vec<TokenTypes>> {
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

/// Attempt to tokenize and parse an input string.
///
/// # Arguments
///
/// * `string` - The string slice that should be tokenized.
///
pub fn parse(string: &str) -> Result<HashMap<Symbol, u32>> {
    // Sanitize any special characters that need to be handled.
    let mut chars: Vec<char> = string.chars().collect();
    sanitize(&mut chars);

    let tokens: Vec<TokenTypes> = tokenizer::tokenize_string(&chars);
    if tokens.is_empty() {
        return Ok(HashMap::new());
    }

    parse_internal(&tokens)
}

/// Attempt to parse a token slice.
///
/// # Arguments
///
/// * `tokens` - A [`TokenTypes`] slice.
///
fn parse_internal(tokens: &[TokenTypes]) -> Result<HashMap<Symbol, u32>> {
    // We have to store the data in this form to allow for
    // term multiplication, see the numeric processing below.
    let mut stack: Vec<Vec<SymbolCounter>> = Vec::new();

    // The segment will be used to apply segment multipliers.
    let mut segment_multiplier = 0;
    let mut segment_start = 0;

    let mut iter = tokens.iter().peekable();

    // Iterate through the tokens.
    while let Some(t) = iter.next() {
        match t {
            TokenTypes::Digit(c) => {
                let mut buffer = String::with_capacity(3);
                buffer.push(*c);

                // Continue until we reach a token of a different type.
                while let Some(TokenTypes::Digit(d)) =
                    iter.next_if(|&x| matches!(x, TokenTypes::Digit(_)))
                {
                    buffer.push(*d);
                }

                let number = parse_number(&buffer);

                // We cannot use 0 as a multiplier within a formula,
                // it's invalid syntax.
                if number == 0 {
                    return Err(ParserError::InvalidMultiplier);
                }

                // Next, we need to apply this multiplier to the last item
                // in the stack. If there is no prior item, then this is an error.
                if let Some(last) = stack.last_mut() {
                    apply_multiplier(last, number);
                } else {
                    // We might be dealing with a formula-specific multiplier.
                    // An example would be calcium sulphate dihydrate: CaSO₄·(H₂O)₂
                    segment_multiplier = number;
                }
            }
            TokenTypes::LParen => {
                // Serialize until we reach the matching bracket.
                let segment = serialize_parenthesis(&mut iter)?;

                // Recursively parse the serialized string.
                let mut parsed = Vec::new();
                for (s, c) in parse_internal(&segment)? {
                    parsed.push(SymbolCounter::new(s, c));
                }
                stack.push(parsed);
            }
            TokenTypes::RParen => {
                return Err(ParserError::MismatchedParenthesis);
            }
            TokenTypes::ElementHead(c) => {
                let mut buffer = String::with_capacity(3);
                buffer.push(*c);

                /*
                  We want to continue until we reach the next token that
                    is -not- part of this symbol.
                  We therefore need to include successive ElementTail entries,
                    but nothing else.
                */
                while let Some(TokenTypes::ElementTail(e)) =
                    iter.next_if(|&x| matches!(x, TokenTypes::ElementTail(_)))
                {
                    buffer.push(*e);
                }

                // Is the symbol valid?
                if let Ok(symbol) = Symbol::from_str(&buffer) {
                    // Create a new element instance.
                    let element = SymbolCounter::new(symbol, 1);
                    stack.push(vec![element]);
                } else {
                    return Err(ParserError::UnrecognizedSymbol);
                }
            }
            TokenTypes::Dot => {
                // We will treat a mid-dot as though it were a bracketed segment.
                // Apply any segment multipliers.
                apply_segment_multiplier(&mut stack[segment_start..], &mut segment_multiplier);
                segment_start = stack.len();

                // Serialize the next data segment.
                let segment = serialize_segment(&mut iter)?;

                let mut parsed = Vec::new();
                for (s, c) in parse_internal(&segment)? {
                    parsed.push(SymbolCounter::new(s, c));
                }
                stack.push(parsed);
            }
            _ => {}
        }
    }

    // A segment multiplier with no segment is an error and the formula is invalid.
    if stack.is_empty() && segment_multiplier > 0 {
        return Err(ParserError::MultiplierNoSegment);
    }

    // Do we have a formula segment multiplier?
    apply_segment_multiplier(&mut stack[segment_start..], &mut segment_multiplier);

    // Now we need to flatten the vector.
    let flat: Vec<SymbolCounter> = stack.iter().flatten().cloned().collect();

    // Finally, we can collect like terms.
    let mut ret: HashMap<Symbol, u32> = HashMap::with_capacity(flat.len());
    for item in flat {
        let e = ret.entry(item.symbol).or_insert(0);
        *e += item.count;
    }

    Ok(ret)
}

fn parse_number(str: &str) -> u32 {
    str.parse::<u32>().unwrap()
}

/// Sanitize an input string.
///
/// # Arguments
///
/// * `chars` - A mutable character slice.
///
fn sanitize(chars: &mut [char]) {
    for c in chars {
        // Subscript digits have to be normalized into their ASCII equivalents.
        let id = *c as u32;
        if (0x2080..=0x2089).contains(&id) {
            let shifted_id = id - 0x2050;
            *c = char::from_u32(shifted_id).unwrap();
        }

        // Mid-dot characters are replaced with periods.
        if *c == '·' {
            *c = '.';
        }
    }
}

#[cfg(test)]
mod tests_parser {
    use crate::{
        definitions::enums::Symbol,
        equation_parser::{error::ParserError, *},
    };

    use std::collections::HashMap;

    struct TestEntryOk<'a> {
        input: &'a str,
        result: HashMap<Symbol, u32>,
    }

    impl TestEntryOk<'_> {
        pub fn new(input: &str, outputs: Vec<(Symbol, u32)>) -> TestEntryOk {
            let mut e = TestEntryOk {
                input,
                result: HashMap::new(),
            };

            for output in outputs {
                e.result.insert(output.0, output.1);
            }

            e
        }
    }

    struct TestEntryErr<'a> {
        input: &'a str,
        error: ParserError,
    }

    impl TestEntryErr<'_> {
        pub fn new(input: &str, error: ParserError) -> TestEntryErr {
            TestEntryErr { input, error }
        }
    }

    #[test]
    fn test_parser_valid_formulae() {
        let tests = [
            // Basic formulae.
            TestEntryOk::new("H", vec![(Symbol::H, 1)]),
            TestEntryOk::new("H2", vec![(Symbol::H, 2)]),
            TestEntryOk::new("H2Ca", vec![(Symbol::H, 2), (Symbol::Ca, 1)]),
            TestEntryOk::new("HCa", vec![(Symbol::H, 1), (Symbol::Ca, 1)]),
            TestEntryOk::new("2HCa", vec![(Symbol::H, 2), (Symbol::Ca, 2)]),
            // Bracketed formulae.
            TestEntryOk::new("(H2Ca2)", vec![(Symbol::H, 2), (Symbol::Ca, 2)]),
            TestEntryOk::new("2(HCa)", vec![(Symbol::H, 2), (Symbol::Ca, 2)]),
            TestEntryOk::new("2(H2Ca)", vec![(Symbol::H, 4), (Symbol::Ca, 2)]),
            TestEntryOk::new("2(H2Ca2)", vec![(Symbol::H, 4), (Symbol::Ca, 4)]),
            TestEntryOk::new("2(H2Ca2)2", vec![(Symbol::H, 8), (Symbol::Ca, 8)]),
            TestEntryOk::new("(H2Ca2)2", vec![(Symbol::H, 4), (Symbol::Ca, 4)]),
            // Segmented formulae.
            TestEntryOk::new(
                "(H2Ca2)·O2",
                vec![(Symbol::H, 2), (Symbol::Ca, 2), (Symbol::O, 2)],
            ),
            TestEntryOk::new(
                "2(H2Ca2)·O2",
                vec![(Symbol::H, 4), (Symbol::Ca, 4), (Symbol::O, 2)],
            ),
            TestEntryOk::new(
                "H2Ca2·O2",
                vec![(Symbol::H, 2), (Symbol::Ca, 2), (Symbol::O, 2)],
            ),
            TestEntryOk::new(
                "H2Ca2·2O2",
                vec![(Symbol::H, 2), (Symbol::Ca, 2), (Symbol::O, 4)],
            ),
            TestEntryOk::new(
                "2H2Ca2·2O2",
                vec![(Symbol::H, 4), (Symbol::Ca, 4), (Symbol::O, 4)],
            ),
            TestEntryOk::new(
                "2H2Ca2·2O2·U2",
                vec![
                    (Symbol::H, 4),
                    (Symbol::Ca, 4),
                    (Symbol::O, 4),
                    (Symbol::U, 2),
                ],
            ),
            TestEntryOk::new("((H2Ca2))", vec![(Symbol::H, 2), (Symbol::Ca, 2)]),
            TestEntryOk::new("((H2)(Ca2))", vec![(Symbol::H, 2), (Symbol::Ca, 2)]),
            // Torture tests.
            TestEntryOk::new(
                "(Zn2(Ca(BrO4))K(Pb)2Rb)3",
                vec![
                    (Symbol::O, 12),
                    (Symbol::K, 3),
                    (Symbol::Ca, 3),
                    (Symbol::Zn, 6),
                    (Symbol::Br, 3),
                    (Symbol::Rb, 3),
                    (Symbol::Pb, 6),
                ],
            ),
            TestEntryOk::new(
                "C228H236F72N12O30P12",
                vec![
                    (Symbol::C, 228),
                    (Symbol::H, 236),
                    (Symbol::F, 72),
                    (Symbol::N, 12),
                    (Symbol::O, 30),
                    (Symbol::P, 12),
                ],
            ),
            // Formulae with subscript unicode characters.
            TestEntryOk::new("H₂", vec![(Symbol::H, 2)]),
            TestEntryOk::new("H₂O2", vec![(Symbol::H, 2), (Symbol::O, 2)]),
        ];

        for (i, test) in tests.into_iter().enumerate() {
            let r = parser::parse(test.input);

            assert!(
                r.is_ok(),
                "Failed to correctly parse valid formulae test {}",
                i
            );

            assert_eq!(
                r.unwrap(),
                test.result,
                "Failed to produce correct output for valid formula test {}",
                i
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_parser_invalid_formulae() {
        // Note, we don't care about the results here as these should fail.
        let tests = [
            // Mismatched brackets.
            TestEntryErr::new("(", ParserError::MismatchedParenthesis),
            TestEntryErr::new("())", ParserError::MismatchedParenthesis),
            TestEntryErr::new("(()", ParserError::MismatchedParenthesis),
            // Invalid segments.
            TestEntryErr::new("·", ParserError::EmptySegment),
            TestEntryErr::new("·H", ParserError::EmptySegment),
            TestEntryErr::new("H·", ParserError::EmptySegment),
            // Multiplier with no terms.
            TestEntryErr::new("2", ParserError::MultiplierNoSegment),
            // Unknown symbol.
            TestEntryErr::new("Zz", ParserError::UnrecognizedSymbol),
            // Invalid multiplier.
            TestEntryErr::new("H0", ParserError::InvalidMultiplier),
            TestEntryErr::new("0(H2)", ParserError::InvalidMultiplier),
        ];

        for (i, test) in tests.into_iter().enumerate() {
            let r = parser::parse(test.input);

            assert_eq!(
                r,
                Err(test.error),
                "Did not receive correct failure error for invalid formula {}",
                i
            );
        }
    }
}
