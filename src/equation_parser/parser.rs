use crate::definitions::element_data;

use std::collections::HashMap;

use super::{
    error::{ParserError, Result},
    parser_result::ParserResult,
    symbol_counter::SymbolCounter,
    tokenizer::{self, TokenTypes},
    utils::*,
};

/// Attempt to tokenize and parse an input string.
///
/// # Arguments
///
/// * `string` - The string slice that should be tokenized.
///
pub fn parse(string: &str) -> Result<ParserResult> {
    // Sanitize any special characters that need to be handled.
    let mut chars: Vec<char> = string.chars().collect();
    sanitize(&mut chars);

    let tokens: Vec<TokenTypes> = tokenizer::tokenize_string(&chars);
    if tokens.is_empty() {
        return Ok(ParserResult::new());
    }

    parse_internal(&tokens)
}

/// Attempt to parse a token slice.
///
/// # Arguments
///
/// * `tokens` - A [`TokenTypes`] slice.
///
fn parse_internal(tokens: &[TokenTypes]) -> Result<ParserResult> {
    // We have to store the data in this form to allow for
    // term multiplication, see the numeric processing below.
    let mut stack: Vec<Vec<SymbolCounter>> = Vec::new();

    // The segment will be used to apply segment multipliers.
    let mut charge: i32 = 0;

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

                // Is the next token a charge sign?
                if let Some(TokenTypes::ChargeSign(c)) = iter.peek() {
                    // Yes, we have to handle a charge here.
                    iter.next();

                    // Charges may only occur at the end of a formula.
                    if let Some(TokenTypes::End) = iter.peek() {
                    } else {
                        return Err(ParserError::InvalidChargePosition);
                    }

                    // The prior digit is the charge for this formula.
                    let charge_digit = buffer.pop().unwrap().to_string();
                    charge = parse_number(&charge_digit) as i32;

                    // We cannot use 0 as a charge, it's invalid syntax.
                    if charge == 0 {
                        return Err(ParserError::InvalidChargeMultiplier);
                    }

                    // If the sign is negative then we need to invert the sign.
                    if *c == '-' {
                        charge *= -1;
                    }

                    // Is there still a multiplier number to parse?
                    if buffer.is_empty() {
                        continue;
                    }
                }

                let number = parse_number(&buffer);

                // We cannot use 0 as a multiplier within a formula,
                // it's invalid syntax.
                if number == 0 {
                    return Err(ParserError::InvalidMultiplier);
                }

                // Next, we need to apply this multiplier to the last item
                // in the stack.
                if let Some(last) = stack.last_mut() {
                    apply_multiplier(last, number);
                } else {
                    return Err(ParserError::MultiplierNoSegment);
                }
            }
            TokenTypes::LParen => {
                // Serialize until we reach the matching bracket.
                let segment = serialize_parenthesis(&mut iter)?;

                // Recursively parse the serialized string.
                let mut parsed = Vec::new();
                for (s, c) in parse_internal(&segment)?.symbols {
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
                if element_data::ELEMENT_DATA.data.contains_key(&buffer) {
                    // Create a new element instance.
                    let element = SymbolCounter::new(buffer, 1);
                    stack.push(vec![element]);
                } else {
                    return Err(ParserError::UnrecognizedSymbol);
                }
            }
            TokenTypes::ChargeSign(s) => {
                // We have encountered a charge sign with no prior digit.
                // This means that the charge is either +1 or -1.

                // If the sign is negative then we need to invert the sign.
                if *s == '-' {
                    charge = -1;
                } else {
                    charge = 1;
                }
            }
            _ => {}
        }
    }

    // Do we have a charge without a formula?
    if charge != 0 && stack.last().is_none() {
        return Err(ParserError::ChargeNoFormula);
    }

    // Now we need to flatten the vector.
    let flat: Vec<SymbolCounter> = stack.iter().flatten().cloned().collect();

    // Finally, we can collect like terms.
    let mut ret: HashMap<String, u32> = HashMap::with_capacity(flat.len());
    for item in flat {
        let e = ret.entry(item.symbol).or_insert(0);
        *e += item.count;
    }

    Ok(ParserResult {
        symbols: ret,
        charge,
    })
}

#[cfg(test)]
mod tests_parser {
    use crate::equation_parser::{error::ParserError, parser_result::ParserResult, *};

    use std::collections::HashMap;

    struct TestEntryOk<'a> {
        input: &'a str,
        result: ParserResult,
    }

    impl TestEntryOk<'_> {
        pub fn new(input: &str, outputs: Vec<(String, u32)>, charge: i32) -> TestEntryOk {
            let mut e = TestEntryOk {
                input,
                result: ParserResult {
                    symbols: HashMap::new(),
                    charge,
                },
            };

            for output in outputs {
                e.result.symbols.insert(output.0, output.1);
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
            TestEntryOk::new("H", vec![("H".to_string(), 1)], 0),
            TestEntryOk::new("H2", vec![("H".to_string(), 2)], 0),
            TestEntryOk::new("H2Ca", vec![("H".to_string(), 2), ("Ca".to_string(), 1)], 0),
            TestEntryOk::new("HCa", vec![("H".to_string(), 1), ("Ca".to_string(), 1)], 0),
            // Bracketed formulae.
            TestEntryOk::new(
                "(H2Ca2)",
                vec![("H".to_string(), 2), ("Ca".to_string(), 2)],
                0,
            ),
            TestEntryOk::new(
                "(H2Ca2)2",
                vec![("H".to_string(), 4), ("Ca".to_string(), 4)],
                0,
            ),
            TestEntryOk::new(
                "((H2Ca2))",
                vec![("H".to_string(), 2), ("Ca".to_string(), 2)],
                0,
            ),
            TestEntryOk::new(
                "((H2)(Ca2))",
                vec![("H".to_string(), 2), ("Ca".to_string(), 2)],
                0,
            ),
            // Torture tests.
            TestEntryOk::new(
                "(Zn2(Ca(BrO4))K(Pb)2Rb)3",
                vec![
                    ("O".to_string(), 12),
                    ("K".to_string(), 3),
                    ("Ca".to_string(), 3),
                    ("Zn".to_string(), 6),
                    ("Br".to_string(), 3),
                    ("Rb".to_string(), 3),
                    ("Pb".to_string(), 6),
                ],
                0,
            ),
            TestEntryOk::new(
                "C228H236F72N12O30P12",
                vec![
                    ("C".to_string(), 228),
                    ("H".to_string(), 236),
                    ("F".to_string(), 72),
                    ("N".to_string(), 12),
                    ("O".to_string(), 30),
                    ("P".to_string(), 12),
                ],
                0,
            ),
            // Formulae with subscript unicode characters.
            TestEntryOk::new("H₂", vec![("H".to_string(), 2)], 0),
            TestEntryOk::new("H₂O2", vec![("H".to_string(), 2), ("O".to_string(), 2)], 0),
            // Formulae with charges
            TestEntryOk::new("[Na]+", vec![("Na".to_string(), 1)], 1),
            TestEntryOk::new("Na+", vec![("Na".to_string(), 1)], 1),
            TestEntryOk::new(
                "[CaCO3]2-",
                vec![
                    ("Ca".to_string(), 1),
                    ("C".to_string(), 1),
                    ("O".to_string(), 3),
                ],
                -2,
            ),
            TestEntryOk::new(
                "CaCO32-",
                vec![
                    ("Ca".to_string(), 1),
                    ("C".to_string(), 1),
                    ("O".to_string(), 3),
                ],
                -2,
            ),
            TestEntryOk::new(
                "CaCO3²⁻",
                vec![
                    ("Ca".to_string(), 1),
                    ("C".to_string(), 1),
                    ("O".to_string(), 3),
                ],
                -2,
            ),
        ];

        for (i, test) in tests.into_iter().enumerate() {
            let r = parser::parse(test.input);

            assert!(
                r.is_ok(),
                "Failed to correctly parse valid formulae test {}, formula: {}",
                i,
                test.input
            );

            assert_eq!(
                r.unwrap(),
                test.result,
                "Failed to produce correct output for valid formula test {}, formula: {}",
                i,
                test.input
            );
        }
    }

    #[test]
    fn test_parser_invalid_formulae() {
        // Note, we don't care about the results here as these should fail.
        let tests = [
            // Mismatched brackets.
            TestEntryErr::new("(", ParserError::MismatchedParenthesis),
            TestEntryErr::new("())", ParserError::MismatchedParenthesis),
            TestEntryErr::new("(()", ParserError::MismatchedParenthesis),
            // Multiplier with no terms.
            TestEntryErr::new("2", ParserError::MultiplierNoSegment),
            // Unknown symbol.
            TestEntryErr::new("Zz", ParserError::UnrecognizedSymbol),
            // Invalid multiplier.
            TestEntryErr::new("H0", ParserError::InvalidMultiplier),
            TestEntryErr::new("0(H2)", ParserError::InvalidMultiplier),
            // Invalid charges.
            TestEntryErr::new("+", ParserError::ChargeNoFormula),
            TestEntryErr::new("2+", ParserError::ChargeNoFormula),
            TestEntryErr::new("2+Na", ParserError::InvalidChargePosition),
            TestEntryErr::new("Ca2+Na", ParserError::InvalidChargePosition),
            TestEntryErr::new("[Na]0+", ParserError::InvalidChargeMultiplier),
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
