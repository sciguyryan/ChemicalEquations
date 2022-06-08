use crate::definitions::enums::Symbol;

use std::{collections::HashMap, iter::Peekable, ops::MulAssign, slice::Iter, str::FromStr};

use super::tokenizer::{self, TokenTypes};

#[derive(Debug, Clone)]
pub struct SymbolCounter {
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

fn serialize_parenthesis(iter: &mut Peekable<Iter<TokenTypes>>) -> Vec<TokenTypes> {
    let mut tokens = Vec::new();

    // The counter to indicate the depth of the parenthesis.
    let mut paren_index: usize = 1;

    // Iterate until we reach the end of the segment.
    // This one is a bit different as we need to locate the matching
    //   parenthesis. If there is a mismatch, we will panic.
    while let Some(t) = iter.peek() {
        match t {
            TokenTypes::LParen => {
                paren_index += 1;
            }
            TokenTypes::RParen => {
                paren_index -= 1;

                // We have found the matching parenthesis.
                if paren_index == 0 {
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

    if paren_index > 0 {
        panic!("Error: mismatching parenthesis.");
    }

    tokens
}

fn serialize_until_segment_end(iter: &mut Peekable<Iter<TokenTypes>>) -> Vec<TokenTypes> {
    let mut tokens = Vec::new();

    // Iterate until we reach the end of the segment.
    while let Some(t) = iter.next_if(|&x| !matches!(x, TokenTypes::Dot)) {
        tokens.push(*t);
    }

    if tokens.is_empty() {
        panic!("An empty segment is not permitted.");
    }

    tokens
}

fn parse_internal(tokens: &[TokenTypes]) -> HashMap<Symbol, u32> {
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
                let mut buffer = String::with_capacity(20);
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
                    panic!("Attempted to apply a zero multiplier");
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
                let segment = serialize_parenthesis(&mut iter);

                // Recursively parse the serialized string.
                let mut paran_parsed = Vec::new();
                for (s, c) in parse_internal(&segment) {
                    paran_parsed.push(SymbolCounter::new(s, c));
                }
                stack.push(paran_parsed);
            }
            TokenTypes::RParen => {
                panic!("Unexpected right parenthesis!");
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
                    panic!("Unrecognized element symbol: {}", &buffer);
                }
            }
            TokenTypes::Dot => {
                // We will treat a mid-dot as though it were a bracketed segment.
                // Apply any segment multipliers.
                apply_segment_multiplier(&mut segment_multiplier, &mut stack[segment_start..]);
                segment_start = stack.len();

                // Serialize the next data segment.
                let segment = serialize_until_segment_end(&mut iter);

                let mut seg_parsed = Vec::new();
                for (s, c) in parse_internal(&segment) {
                    seg_parsed.push(SymbolCounter::new(s, c));
                }
                stack.push(seg_parsed);
            }
            _ => {}
        }
    }

    // TODO: decide if I should warn when having an empty stack
    // TODO: with a multiplier applied.
    if stack.is_empty() && segment_multiplier > 0 {
        panic!("Segment multiplier applied with no segment.");
    }

    // Do we have a formula segment multiplier?
    apply_segment_multiplier(&mut segment_multiplier, &mut stack[segment_start..]);

    // Now we need to flatten the vector.
    let flat: Vec<SymbolCounter> = stack.iter().flatten().cloned().collect();

    // Finally, we can collect like terms.
    let mut ret: HashMap<Symbol, u32> = HashMap::with_capacity(flat.len());
    for item in flat {
        let e = ret.entry(item.symbol).or_insert(0);
        *e += item.count;
    }

    ret
}

pub fn parse(string: &str) -> HashMap<Symbol, u32> {
    // Sanitize any special characters that need to be handled.
    let mut chars: Vec<char> = string.chars().collect();
    sanitize(&mut chars);

    let tokens: Vec<TokenTypes> = tokenizer::tokenize_string(&chars);
    let len = tokens.len();
    if len == 0 {
        return HashMap::new();
    }

    parse_internal(&tokens)
}

fn apply_segment_multiplier(mul: &mut u32, stack: &mut [Vec<SymbolCounter>]) {
    // Do we have a formula segment multiplier?
    if *mul > 0 {
        // Apply the segment multiplier to the segment.
        for segment in stack {
            apply_multiplier(segment, *mul);
        }

        *mul = 0;
    }
}

fn apply_multiplier(slice: &mut [SymbolCounter], constant: u32) {
    for s in slice {
        *s *= constant;
    }
}

fn sanitize(chars: &mut [char]) {
    for c in chars {
        // Subscript digits have to be normalized into their ASCII equivalents.
        let id = *c as u32;
        if (0x2080..=0x2089).contains(&id) {
            let shifted_id = id - 0x2050;
            *c = char::from_u32(shifted_id).unwrap();
        }

        if *c == '·' {
            *c = '.';
        }
    }
}

fn parse_number(str: &str) -> u32 {
    str.parse::<u32>().unwrap()
}

#[cfg(test)]
mod tests_parser {
    use crate::{definitions::enums::Symbol, equation_parser::*};

    use std::collections::HashMap;

    struct TestEntry<'a> {
        input: &'a str,
        result: HashMap<Symbol, u32>,
    }

    impl TestEntry<'_> {
        pub fn new(input: &str, outputs: Vec<(Symbol, u32)>) -> TestEntry {
            let mut e = TestEntry {
                input,
                result: HashMap::new(),
            };

            for output in outputs {
                e.result.insert(output.0, output.1);
            }

            e
        }
    }

    #[test]
    fn test_parser_valid_formulae() {
        let tests = [
            // Basic formulae.
            TestEntry::new("H", vec![(Symbol::H, 1)]),
            TestEntry::new("H2", vec![(Symbol::H, 2)]),
            TestEntry::new("H2Ca", vec![(Symbol::H, 2), (Symbol::Ca, 1)]),
            TestEntry::new("HCa", vec![(Symbol::H, 1), (Symbol::Ca, 1)]),
            TestEntry::new("2HCa", vec![(Symbol::H, 2), (Symbol::Ca, 2)]),
            // Bracketed formulae.
            TestEntry::new("(H2Ca2)", vec![(Symbol::H, 2), (Symbol::Ca, 2)]),
            TestEntry::new("2(HCa)", vec![(Symbol::H, 2), (Symbol::Ca, 2)]),
            TestEntry::new("2(H2Ca)", vec![(Symbol::H, 4), (Symbol::Ca, 2)]),
            TestEntry::new("2(H2Ca2)", vec![(Symbol::H, 4), (Symbol::Ca, 4)]),
            TestEntry::new("2(H2Ca2)2", vec![(Symbol::H, 8), (Symbol::Ca, 8)]),
            TestEntry::new("(H2Ca2)2", vec![(Symbol::H, 4), (Symbol::Ca, 4)]),
            // Segmented formulae.
            TestEntry::new(
                "(H2Ca2)·O2",
                vec![(Symbol::H, 2), (Symbol::Ca, 2), (Symbol::O, 2)],
            ),
            TestEntry::new(
                "2(H2Ca2)·O2",
                vec![(Symbol::H, 4), (Symbol::Ca, 4), (Symbol::O, 2)],
            ),
            TestEntry::new(
                "H2Ca2·O2",
                vec![(Symbol::H, 2), (Symbol::Ca, 2), (Symbol::O, 2)],
            ),
            TestEntry::new(
                "H2Ca2·2O2",
                vec![(Symbol::H, 2), (Symbol::Ca, 2), (Symbol::O, 4)],
            ),
            TestEntry::new(
                "2H2Ca2·2O2",
                vec![(Symbol::H, 4), (Symbol::Ca, 4), (Symbol::O, 4)],
            ),
            TestEntry::new(
                "2H2Ca2·2O2·U2",
                vec![
                    (Symbol::H, 4),
                    (Symbol::Ca, 4),
                    (Symbol::O, 4),
                    (Symbol::U, 2),
                ],
            ),
            TestEntry::new("((H2Ca2))", vec![(Symbol::H, 2), (Symbol::Ca, 2)]),
            TestEntry::new("((H2)(Ca2))", vec![(Symbol::H, 2), (Symbol::Ca, 2)]),
            // Torture tests.
            TestEntry::new(
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
            TestEntry::new(
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
            TestEntry::new("H₂", vec![(Symbol::H, 2)]),
            TestEntry::new("H₂O2", vec![(Symbol::H, 2), (Symbol::O, 2)]),
        ];

        for (i, test) in tests.into_iter().enumerate() {
            let r = parser_v3::parse(test.input);

            assert_eq!(
                r, test.result,
                "Failed to correctly parse valid formulae test {}",
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
            TestEntry::new("(", vec![]),
            TestEntry::new("())", vec![]),
            TestEntry::new("(()", vec![]),
            // Invalid segments.
            TestEntry::new("·", vec![]),
            TestEntry::new("·H", vec![]),
            TestEntry::new("H·", vec![]),
            // Multiplier with no terms.
            TestEntry::new("2", vec![]),
            // Invalid symbol.
            TestEntry::new("Zz", vec![]),
        ];

        for (i, test) in tests.into_iter().enumerate() {
            let result = std::panic::catch_unwind(|| parser_v3::parse(test.input));
            assert!(
                result.is_err(),
                "Failed to panic with invalid formulae test {}",
                i
            );
        }
    }
}
