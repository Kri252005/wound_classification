export type Probability = {
  class: string
  class_index: number
  probability: number
}

export type PredictionResponse = {
  analysis_id: string

  prediction: {
    class: string
    class_index: number
    confidence: number
  }

  probabilities: Probability[]

  model: {
    name: string
    version: string
  }

  processing_time_ms: number
}
