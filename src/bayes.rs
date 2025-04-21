use serde::{de::Error, Deserialize, Serialize};
use std::{collections::HashMap, io::Read, vec};

/// Base trait for all Naïve Bayes classifiers
pub trait NaiveBayesClassifier {
    fn fit(&mut self, tokens: &[usize], label: usize);
    fn predict(&self, tokens: &[usize]) -> usize;
    fn predict_probas(&self, tokens: &[usize]) -> Box<[f64]>;
}

/// A Naive Bayes classifier using binary features (presence or absence of a specific word).
#[derive(Serialize, Deserialize)]
pub struct BernouliNB {
    /// Feature counts for each label.
    feature_counts: Box<[Box<[usize]>]>,
    /// Total number of samples
    total_samples: usize,
    /// Count of target labels
    target_counts: Box<[usize]>,
    /// The Laplace smoothing factor
    laplace_factor: f64,
}

impl BernouliNB {
    pub fn new(n_features: usize, n_labels: usize, laplace_smoothing: f64) -> Self {
        Self {
            feature_counts: vec![vec![0; n_features].into_boxed_slice(); n_labels]
                .into_boxed_slice(),
            total_samples: 0,
            target_counts: vec![0; n_labels].into_boxed_slice(),
            laplace_factor: laplace_smoothing,
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
}

impl NaiveBayesClassifier for BernouliNB {
    /// Fits the classifier on the specified tokenized text.
    fn fit(&mut self, tokens: &[usize], label: usize) {
        assert!(label < self.target_counts.len());

        for &token in tokens {
            self.feature_counts[label][token] += 1;
        }

        self.total_samples += 1;
        self.target_counts[label] += 1;
    }

    /// Predicts the target label for the sparse tokenized text
    fn predict(&self, tokens: &[usize]) -> usize {
        self.predict_probas(tokens)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(tgt, _)| tgt)
            .unwrap()
    }

    /// Returns the target label probabilities for the sparse tokenized text
    fn predict_probas(&self, tokens: &[usize]) -> Box<[f64]> {
        self.target_counts
            .iter()
            .enumerate()
            .map(|(tgt, &count)| {
                let mut prob = 0.0;
                for &token in tokens {
                    prob += ((self.feature_counts[tgt][token] as f64 + self.laplace_factor)
                        / (count as f64 + self.target_counts.len() as f64 * self.laplace_factor))
                        .ln();
                }
                prob += ((count as f64 + self.laplace_factor)
                    / (self.total_samples as f64 + 2.0 * self.laplace_factor))
                    .ln();
                prob
            })
            .collect::<Vec<f64>>()
            .into_boxed_slice()
    }
}

/// A Naive Bayes classifier using multinomial features (word frequency).
#[derive(Serialize, Deserialize)]
pub struct MultinomialNB {
    /// Feature counts for each label.
    feature_counts: Box<[Box<[usize]>]>,
    /// Total feature counts per label.
    label_feature_totals: Box<[usize]>,
    /// Count of target labels.
    target_counts: Box<[usize]>,
    /// The Laplace smoothing factor.
    laplace_factor: f64,
    /// Total number of samples.
    total_samples: usize,
}

impl MultinomialNB {
    pub fn new(n_features: usize, n_labels: usize, laplace_smoothing: f64) -> Self {
        Self {
            feature_counts: vec![vec![0; n_features].into_boxed_slice(); n_labels]
                .into_boxed_slice(),
            label_feature_totals: vec![0; n_labels].into_boxed_slice(),
            total_samples: 0,
            target_counts: vec![0; n_labels].into_boxed_slice(),
            laplace_factor: laplace_smoothing,
        }
    }
}

impl NaiveBayesClassifier for MultinomialNB {
    /// Predicts the target label for the tokenized text
    fn predict(&self, tokens: &[usize]) -> usize {
        self.predict_probas(tokens)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(tgt, _)| tgt)
            .unwrap()
    }

    /// Fits the classifier on the specified tokenized text.
    fn fit(&mut self, tokens: &[usize], target: usize) {
        assert!(target < self.target_counts.len());
        tokens
            .iter()
            .copied()
            .fold(HashMap::new(), |mut map, val| {
                map.entry(val).and_modify(|frq| *frq += 1).or_insert(1usize);
                map
            })
            .into_iter()
            .for_each(|(token, count)| {
                self.feature_counts[target][token] += count;
                self.label_feature_totals[target] += count;
            });

        self.total_samples += 1;
        self.target_counts[target] += 1;
    }

    /// Returns the target label probabilities for the tokenized text
    fn predict_probas(&self, tokens: &[usize]) -> Box<[f64]> {
        let token_map = tokens.iter().copied().fold(HashMap::new(), |mut map, val| {
            map.entry(val).and_modify(|frq| *frq += 1).or_insert(1usize);
            map
        });
        let n_features = self.feature_counts[0].len();

        self.target_counts
            .iter()
            .enumerate()
            .map(|(tgt, &count)| {
                let prior = (count as f64 + self.laplace_factor)
                    / (self.total_samples as f64
                        + self.target_counts.len() as f64 * self.laplace_factor);

                let mut log_prob = prior.ln();

                for (token, &token_count) in &token_map {
                    if *token >= n_features {
                        continue;
                    }

                    let feature_count = self.feature_counts[tgt][*token] as f64;
                    let total_features = self.label_feature_totals[tgt] as f64;

                    let token_prob = (feature_count + self.laplace_factor)
                        / (total_features + n_features as f64 * self.laplace_factor);

                    log_prob += token_count as f64 * token_prob.ln();
                }

                log_prob.exp()
            })
            .collect::<Vec<f64>>()
            .into_boxed_slice()
    }
}
