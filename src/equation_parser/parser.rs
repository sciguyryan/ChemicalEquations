use crate::definitions::enums::Symbol;

use std::{collections::HashMap, ops::MulAssign, str::FromStr};

fn is_subscript_digit(char: char) -> bool {
    let id = char as u32;
    (0x2080..=0x2089).contains(&id)
}

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

pub fn parse(string: &str) -> HashMap<Symbol, u32> {
    let mut stack: Vec<Vec<SymbolCounter>> = Vec::new();

    let mut chars: Vec<char> = string.chars().collect();
    let len = chars.len();

    // Sanitize any special characters that need to be handled.
    sanitize(&mut chars);

    let mut cursor = 0;
    while cursor < len {
        // Get the character at the current position.
        match chars[cursor] {
            '(' => {
                //println!("Bracket found at position: {}", cursor);
                // If we have a matching bracket then we will recursively pass the substring
                // to ourselves and parse that.
                if let Some(i) = &chars[cursor..].iter().position(|c| *c == ')') {
                    // Move past the opening parenthesis.
                    cursor += 1;

                    // The end should be at the character directly before the closing
                    // bracket.
                    let end = cursor + i - 1;
                    //println!("Closing bracket found at position: {}", end);

                    // Push the sub-entries onto the vector.
                    //println!("End = {}", &string[cursor..end]);
                    let sub_stack: Vec<SymbolCounter> = parse(&string[cursor..end])
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
                //println!("Number found at position: {}", cursor);
                // We want to look for the next item that is not a number.
                let end = if let Some(i) = &chars[cursor..].iter().position(|c| !c.is_ascii_digit())
                {
                    cursor + *i
                } else {
                    len
                };
                //println!("Number terminator found at position: {}", end);

                let number = parse_number(&chars[cursor..end]);
                //println!("Number = {:?}", number);

                // Next, we need to apply this multiplier to the last item
                // in the stack. If there is no prior item, then this is an error.
                if let Some(last) = stack.last_mut() {
                    apply_multiplier(last, number);
                } else {
                    panic!("Error: numeric multiplier with no prior term.");
                }

                cursor = end;
            }
            'A'..='Z' => {
                //println!("Symbol found at position: {}", cursor);

                // We want to look for the next item that is not a lowercase
                // character.
                // We always want ensure that the final character is included
                // in the slice.
                let end = if let Some(i) =
                    &chars[cursor..].iter().position(|c| !c.is_ascii_lowercase())
                {
                    cursor + *i
                } else {
                    len
                } + 1;
                //println!("Symbol terminator found at position: {}", end);

                let symbol_slice = &chars[cursor..end];
                //println!("Symbol = {:?}", symbol_slice);

                // Next, we need to try and parse the symbol into a Symbols enum
                // item.
                let s = symbol_slice.iter().cloned().collect::<String>();
                if let Ok(symbol) = Symbol::from_str(&s) {
                    //println!("Symbol (enum) = {:?}", symbol);

                    // Create a new element instance.
                    let element = SymbolCounter::new(symbol, 1);
                    stack.push(vec![element]);
                } else {
                    panic!("Unrecognized element symbol: {}", s);
                }

                cursor = end;
            }
            '·' | '.' => {
                // These can be found in some formulae, they can be skipped.
                cursor += 1;
                continue;
            }
            _ => {
                println!("Invalid character, {}, at index {}", chars[cursor], cursor);
                cursor += 1;
            }
        }
    }

    // Not we need to flatten the vector.
    let flat: Vec<SymbolCounter> = stack.iter().flatten().cloned().collect();
    //println!("{:?}", flat);

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
        // Subscript digits have to be handled separately.
        if is_subscript_digit(*c) {
            let shifted_id = (*c as u32) - 0x2050;
            *c = char::from_u32(shifted_id).unwrap();
        }
    }
}

fn parse_number(chars: &[char]) -> u32 {
    let str: String = chars.iter().collect();
    str.parse::<u32>().unwrap()
}
