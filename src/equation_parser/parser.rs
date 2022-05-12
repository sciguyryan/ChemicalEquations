use crate::definitions::enums::Symbol;

use std::{collections::HashMap, ops::MulAssign, str::FromStr};

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

enum TokenTypes {
    /// Initial token.
    Start,
    /// A numeric character.
    Number,
    /// A left (opening) parenthesis.
    LParan,
    /// A right (closing) parenthesis.
    RParan,
    /// The start of an element symbol.
    ElementHead,
    /// The (optional) continuation of an element symbol.
    ElementTail,
    /// A middot special character.
    Dot,
    /// The end of the token stream.
    End,
}

pub fn parse2(string: &str) -> HashMap<Symbol, u32> {
    // We have to store the data in this form to allow for
    // term multiplication, see the numeric processing below.
    let mut stack: Vec<Vec<SymbolCounter>> = Vec::new();

    let mut chars: Vec<char> = string.chars().collect();
    let len = chars.len();

    // Sanitize any special characters that need to be handled.
    sanitize(&mut chars);

    // Not we need to flatten the vector.
    let flat: Vec<SymbolCounter> = stack.iter().flatten().cloned().collect();

    // Finally, we can collect like terms.
    let mut ret: HashMap<Symbol, u32> = HashMap::new();
    for item in flat {
        let e = ret.entry(item.symbol).or_insert(0);
        *e += item.count;
    }

    ret
}

pub fn parse(string: &str) -> HashMap<Symbol, u32> {
    // We have to store the data in this form to allow for
    // term multiplication, see the numeric processing below.
    let mut stack: Vec<Vec<SymbolCounter>> = Vec::new();

    let mut chars: Vec<char> = string.chars().collect();
    let len = chars.len();

    // Sanitize any special characters that need to be handled.
    sanitize(&mut chars);

    // This will indicate that we are at the end of a segment of
    // a formula.
    let mut end_of_segment = false;
    let mut segment_multiplier = 0;
    let mut segment_start_index = 0;
    let mut segment_end_index = 0;

    let mut cursor = 0;
    while cursor < len {
        // Get the character at the current position.
        match chars[cursor] {
            '(' => {
                // If we have a matching bracket then we will recursively pass the substring
                // to ourselves and parse that.
                if let Some(i) = &chars[cursor..].iter().position(|c| *c == ')') {
                    // Move past the opening parenthesis.
                    cursor += 1;

                    // The end should be at the character directly before the closing
                    // bracket.
                    let end = cursor + i - 1;

                    // Push the sub-entries onto the vector.
                    let str = &chars[cursor..end].iter().cloned().collect::<String>();
                    println!("{}", str);
                    let sub_stack: Vec<SymbolCounter> = parse(str)
                        .into_iter()
                        .map(|(s, c)| SymbolCounter::new(s, c))
                        .collect();

                    stack.push(sub_stack);

                    // Move past the closing parenthesis.
                    cursor = end + 1;
                } else {
                    panic!("Error: mismatching parenthesis.");
                }
            }
            '0'..='9' => {
                // We want to look for the next item that is not a number.
                let end = if let Some(i) = &chars[cursor..].iter().position(|c| !c.is_ascii_digit())
                {
                    cursor + *i
                } else {
                    len
                };

                let number = parse_number(&chars[cursor..end]);

                // We cannot use 0 as a multiplier within a formula, it's invalid syntax.
                if number == 0 {
                    panic!("Attempted to apply a zero multiplier");
                }

                // Next, we need to apply this multiplier to the last item
                // in the stack. If there is no prior item, then this is an error.
                if let Some(last) = stack.last_mut() {
                    apply_multiplier(last, number);
                } else {
                    //panic!("Error: numeric multiplier with no prior term.");
                    // We might be dealing with a formula-specific multiplier.
                    // An example would be calcium sulphate dihydrate: CaSO₄·(H₂O)₂
                    segment_multiplier = number;
                }

                cursor = end;
            }
            'A'..='Z' => {
                let start = cursor;

                // Move past the opening parenthesis.
                cursor += 1;

                // We want to look for the next item that is not a lowercase
                // character.
                // We always want ensure that the final character is included
                // in the slice.
                let end = if let Some(i) =
                    &chars[cursor..].iter().position(|c| !c.is_ascii_lowercase())
                {
                    cursor + *i
                } else {
                    len - 1
                };

                let symbol_slice = &chars[start..end];
                println!("symbol_slice = {:?}", symbol_slice);

                // Next, we need to try and parse the symbol into a Symbols enum
                // item.
                let s = symbol_slice.iter().cloned().collect::<String>();
                if let Ok(symbol) = Symbol::from_str(&s) {
                    // Create a new element instance.
                    let element = SymbolCounter::new(symbol, 1);
                    stack.push(vec![element]);
                } else {
                    panic!("Unrecognized element symbol: {}", s);
                }

                cursor += symbol_slice.len() - 1;
            }
            '.' => {
                // We have reached the end of a formula segment.
                end_of_segment = true;

                // The end of this segment is the length of the vector.
                segment_end_index = stack.len();

                // These can be found in some formulae, they can be skipped.
                cursor += 1;
            }
            _ => {
                panic!("Invalid character, {}, at index {}", chars[cursor], cursor);
                //cursor += 1;
            }
        }

        // If we are at the end of the string, the segment end will be at
        // whatever the length of the vector currently is.
        if cursor == len {
            segment_end_index = stack.len() - 1;
            end_of_segment = true;
        }

        // Do we have a formula segment multiplier?
        if segment_multiplier > 0 && end_of_segment {
            // Apply the segment multiplier to the segment.
            for i in stack[segment_start_index..=segment_end_index].iter_mut() {
                apply_multiplier(i, segment_multiplier);
            }

            segment_multiplier = 0;
            segment_start_index = segment_end_index;
        }

        // This would be invalid syntax, and would result from something like this:
        // 2
        // CaSO₄·2
        if segment_multiplier > 0 && cursor == len {
            panic!(
                "Segment multiplier applied with no segment. {} {}",
                cursor, len
            );
        }
    }

    // Not we need to flatten the vector.
    let flat: Vec<SymbolCounter> = stack.iter().flatten().cloned().collect();

    // Finally, we can collect like terms.
    let mut ret: HashMap<Symbol, u32> = HashMap::new();
    for item in flat {
        let e = ret.entry(item.symbol).or_insert(0);
        *e += item.count;
    }

    ret
}

fn apply_multiplier(vec: &mut [SymbolCounter], constant: u32) {
    for v in vec.iter_mut() {
        *v *= constant;
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

fn parse_number(chars: &[char]) -> u32 {
    let str: String = chars.iter().collect();
    str.parse::<u32>().unwrap()
}
