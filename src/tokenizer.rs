use indexmap::IndexSet;
use regex::Regex;
use serde::{de::Error, Deserialize, Serialize};
use std::{borrow::Cow, io::Read};

/// A text tokenizer
#[derive(Debug, Serialize, Deserialize)]
pub struct Tokenizer {
    /// The tokenizer vocabulary dictionnary.
    dict: IndexSet<Box<str>>,
    /// The set of punctuation characters to normalize.
    punct: Box<str>,
}

impl Tokenizer {
    pub fn new(punct: &str) -> Self {
        Self {
            dict: IndexSet::new(),
            punct: punct.into(),
        }
    }

    /// Loads a tokenizer from a file.
    pub fn load_from_file(file: &mut dyn Read) -> Result<Self, serde_json::Error> {
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)
            .map_err(serde_json::Error::custom)?;
        serde_json::from_str(&buffer)
    }

    /// Saves a tokenizer to a file.
    pub fn save_to_file(&self, file: &mut dyn std::io::Write) -> Result<(), serde_json::Error> {
        let serialized = serde_json::to_string(self)?;
        file.write_all(serialized.as_bytes())
            .map_err(serde_json::Error::custom)
    }

    /// Fits the tokenizer on the provided text and returns the tokens of the text.
    pub fn fit(&mut self, text: &str) -> Vec<usize> {
        let normalized = self.normalize(text);
        normalized
            .split_whitespace()
            .map(|w| self.dict.insert_full(w.into()))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Tokenize the supplied text into a list of tokens.
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let normalized = self.normalize(text);
        normalized
            .split_whitespace()
            .filter_map(|w| self.dict.get_index_of(w))
            .collect()
    }

    /// Tokenize the supplied text into a vector representing the presence of words.
    pub fn tokenize_sparse(&self, text: &str) -> Vec<usize> {
        let normalized = self.normalize(text);
        let mut tokens: Vec<usize> = normalized
            .split_whitespace()
            .filter_map(|w| self.dict.get_index_of(w))
            .collect();

        tokens.sort_unstable();
        tokens.dedup();

        tokens
    }

    /// Normalize punctuation in the passed in text.
    pub fn normalize(&self, text: &str) -> Box<str> {
        let re = Regex::new(&self.punct).unwrap();
        let result: Cow<'_, str> = re.replace_all(text, " $1 ");
        let collapsed = Regex::new(r"\s{2,}").unwrap().replace_all(&result, " ");
        collapsed.trim().into()
    }

    /// The total count of tokens
    #[inline]
    pub fn token_count(&self) -> usize {
        self.dict.len()
    }
}
