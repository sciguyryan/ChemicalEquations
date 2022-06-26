use super::{
    parser::tokenizer::{self, *},
    utils::*,
};

/// Tokenize and format an input formula string.
///
/// # Arguments
///
/// * `string` - The string slice that should be tokenized.
///
/// `Note` This this method does `not` completely validate the correctness of a chemical formula.
///
pub fn format(string: &str) -> String {
    // Sanitize any special characters that need to be handled.
    let mut chars: Vec<char> = string.chars().collect();
    sanitize(&mut chars);

    let tokens: Vec<TokenTypes> = tokenizer::tokenize_string(&chars);
    if tokens.is_empty() {
        return String::new();
    }

    format_internal(&tokens)
}

/// Create a formatted formula string from a token slice.
///
/// # Arguments
///
/// * `tokens` - A [`TokenTypes`] slice.
///
/// `Note` This this method does `not` completely validate the correctness of a chemical formula.
///
fn format_internal(tokens: &[TokenTypes]) -> String {
    let mut formatted = String::with_capacity(tokens.len());

    let mut iter = tokens.iter().peekable();

    // Iterate through the tokens.
    while let Some(t) = iter.next() {
        match t {
            TokenTypes::Digit(c) => {
                let mut buffer = String::with_capacity(3);
                buffer.push(*c);

                let mut charge_buff = String::with_capacity(3);

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

                    // The prior digit is the charge for this formula.
                    // We assume that only one digit will be relevant for the charge since I don't know
                    // of any charges that are 10 or above. Should one come to my attention then I will
                    // come up with a different solution.
                    let charge_digit = buffer.pop().unwrap();

                    // Push the charge and the charge sign, then the charge multiplier.
                    charge_buff.push(map_to_superscript(charge_digit));
                    charge_buff.push(map_to_superscript(*c));
                }

                for d in buffer.chars() {
                    formatted.push(map_to_subscript(d));
                }

                for d in charge_buff.chars() {
                    formatted.push(d);
                }
            }
            TokenTypes::LParen(c) => {
                formatted.push(*c);
            }
            TokenTypes::RParen(c) => {
                formatted.push(*c);
            }
            TokenTypes::ElementHead(c) => {
                formatted.push(*c);
            }
            TokenTypes::ElementTail(c) => {
                formatted.push(*c);
            }
            TokenTypes::ChargeSign(c) => {
                formatted.push(map_to_superscript(*c));
            }
            _ => {}
        }
    }

    formatted
}

#[cfg(test)]
mod tests_formatter {
    use crate::formula_tools::formatter;

    struct TestEntryOk<'a> {
        input: &'a str,
        result: &'a str,
    }

    impl<'a> TestEntryOk<'a> {
        pub fn new(input: &'a str, result: &'a str) -> TestEntryOk<'a> {
            Self { input, result }
        }
    }

    #[test]
    fn test_formatter() {
        let tests = [
            // Basic formulae.
            TestEntryOk::new("H", "H"),
            TestEntryOk::new("H2", "H₂"),
            TestEntryOk::new("H2Ca", "H₂Ca"),
            TestEntryOk::new("HCa", "HCa"),
            // Bracketed formulae.
            TestEntryOk::new("(H2Ca2)", "(H₂Ca₂)"),
            TestEntryOk::new("(H2Ca2)2", "(H₂Ca₂)₂"),
            TestEntryOk::new("((H2Ca2))", "((H₂Ca₂))"),
            TestEntryOk::new("((H2)(Ca2))", "((H₂)(Ca₂))"),
            // Torture tests.
            TestEntryOk::new("(Zn2(Ca(BrO4))K(Pb)2Rb)3", "(Zn₂(Ca(BrO₄))K(Pb)₂Rb)₃"),
            TestEntryOk::new("C228H236F72N12O30P12", "C₂₂₈H₂₃₆F₇₂N₁₂O₃₀P₁₂"),
            TestEntryOk::new(
                "[(Zn2(Ca(BrO4))K(Pb)2Rb)3]²⁺",
                "[(Zn₂(Ca(BrO₄))K(Pb)₂Rb)₃]²⁺",
            ),
            // Formulae with subscript and superscript unicode characters.
            TestEntryOk::new("H₂", "H₂"),
            TestEntryOk::new("H₂O2", "H₂O₂"),
            TestEntryOk::new("H²O²", "H₂O₂"),
            // Formulae with charges
            TestEntryOk::new("[Na]+", "[Na]⁺"),
            TestEntryOk::new("Na+", "Na⁺"),
            TestEntryOk::new("[CaCO3]2-", "[CaCO₃]²⁻"),
            TestEntryOk::new("CaCO32-", "CaCO₃²⁻"),
            TestEntryOk::new("CaCO3²⁺", "CaCO₃²⁺"),
            TestEntryOk::new("CaCO3²+", "CaCO₃²⁺"),
        ];

        for (i, test) in tests.into_iter().enumerate() {
            let r = formatter::format(test.input);

            assert_eq!(
                r, test.result,
                "Failed to produce correct output for valid formula test {}, formula: {}",
                i, test.input
            );
        }
    }
}
