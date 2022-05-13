use crate::definitions::enums::Symbol;

use std::{collections::HashMap, iter::Peekable, ops::MulAssign, slice::Iter, str::FromStr};

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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TokenTypes {
    /// Initial token.
    Start,
    /// A numeric character.
    Digit(char),
    /// A left (opening) parenthesis.
    LParen,
    /// A right (closing) parenthesis.
    RParen,
    /// The characters of an element symbol.
    ElementHead(char),
    /// The characters of an element symbol.
    ElementTail(char),
    /// A mid-dot special character.
    Dot,
    /// The end token.
    End,
}

pub fn lex_string(chars: &[char]) -> Vec<TokenTypes> {
    let mut tokens: Vec<TokenTypes> = vec![TokenTypes::Start];

    for (i, c) in chars.iter().enumerate() {
        match c {
            '(' => {
                tokens.push(TokenTypes::LParen);
            }
            ')' => {
                tokens.push(TokenTypes::RParen);
            }
            '0'..='9' => {
                tokens.push(TokenTypes::Digit(*c));
            }
            'A'..='Z' => {
                tokens.push(TokenTypes::ElementHead(*c));
            }
            'a'..='z' => {
                tokens.push(TokenTypes::ElementTail(*c));
            }
            '.' => {
                tokens.push(TokenTypes::Dot);
            }
            _ => {
                eprintln!("Invalid character, {}, at index {}", c, i);
            }
        }
    }

    tokens.push(TokenTypes::End);

    tokens
}

fn serialize_until_matching_paren(buffer: &mut String, iter: &mut Peekable<Iter<TokenTypes>>) {
    let mut paren_index: usize = 1;

    // Iterate until we reach the end of the segment.
    // This one is a bit different as we need to locate the matching
    //   parenthesis. If there is a mismatch, we will panic.
    while let Some(t) = iter.peek() {
        match t {
            TokenTypes::Digit(d) => {
                buffer.push(*d);
            }
            TokenTypes::LParen => {
                buffer.push('(');
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

                buffer.push(')');
            }
            TokenTypes::ElementHead(e) => {
                buffer.push(*e);
            }
            TokenTypes::ElementTail(e) => {
                buffer.push(*e);
            }
            TokenTypes::Dot => {
                buffer.push('.');
            }
            _ => {}
        }

        iter.next();
    }

    if paren_index > 0 {
        panic!("Error: mismatching parenthesis.");
    }
}

fn serialize_until_segment_end(buffer: &mut String, iter: &mut Peekable<Iter<TokenTypes>>) {
    // Iterate until we reach the end of the segment.
    while let Some(t) = iter.next_if(|&x| !matches!(x, TokenTypes::Dot)) {
        match t {
            TokenTypes::Digit(d) => {
                buffer.push(*d);
            }
            TokenTypes::LParen => {
                buffer.push('(');
            }
            TokenTypes::RParen => {
                buffer.push(')');
            }
            TokenTypes::ElementHead(e) => {
                buffer.push(*e);
            }
            TokenTypes::ElementTail(e) => {
                buffer.push(*e);
            }
            TokenTypes::Dot => {
                // This will be the start of a new segment.
                break;
            }
            _ => {}
        }
    }

    if buffer.is_empty() {
        panic!("An empty segment is not permitted.");
    }
}

pub fn parse2(string: &str) -> HashMap<Symbol, u32> {
    // We have to store the data in this form to allow for
    // term multiplication, see the numeric processing below.
    let mut stack: Vec<Vec<SymbolCounter>> = Vec::new();

    let mut chars: Vec<char> = string.chars().collect();

    // Sanitize any special characters that need to be handled.
    sanitize(&mut chars);

    let tokens: Vec<TokenTypes> = lex_string(&chars);
    let len = tokens.len();
    if len == 0 {
        return HashMap::new();
    }

    //eprintln!("{:?}", tokens);

    let mut buffer = String::new();

    // The segment will be used to apply segment multipliers.
    let mut segment_multiplier = 0;
    let mut segment_start = 0;

    let mut iter = tokens.iter().peekable();

    // Iterate through the tokens.
    while let Some(t) = iter.next() {
        match t {
            TokenTypes::Digit(c) => {
                buffer.clear();

                buffer.push(*c);

                // Consume until we reach a token of a different type.
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
                buffer.clear();

                // Serialize the next data segment.
                serialize_until_matching_paren(&mut buffer, &mut iter);

                let mut paran_parsed = Vec::new();
                for (s, c) in parse2(&buffer) {
                    paran_parsed.push(SymbolCounter::new(s, c));
                }
                stack.push(paran_parsed);
            }
            TokenTypes::RParen => {
                eprintln!("Unexpected right parenthesis!");
            }
            TokenTypes::ElementHead(c) => {
                buffer.clear();

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
                buffer.clear();

                // Apply any segment multipliers.
                apply_segment_multiplier(&mut segment_multiplier, &mut stack[segment_start..]);
                segment_start = stack.len();

                // Serialize the next data segment.
                serialize_until_segment_end(&mut buffer, &mut iter);

                let mut seg_parsed = Vec::new();
                for (s, c) in parse2(&buffer) {
                    seg_parsed.push(SymbolCounter::new(s, c));
                }
                stack.push(seg_parsed);
            }
            _ => {}
        }
    }

    // TODO: decide if I should warn when having an empty stack
    // TODO: with a multiplier applied.

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
