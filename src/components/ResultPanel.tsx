import { ScanLine } from 'lucide-react'
import ConfidenceBars from './ConfidenceBars'
import type { PredictionResponse } from '../types/prediction'

const CONFIDENCE_THRESHOLD = 0.70

type ResultPanelProps = {
  result: PredictionResponse | null
  status: 'idle' | 'running' | 'error'
}

function ResultPanel({ result, status }: ResultPanelProps) {
  return (
    <section className="result-panel">
      <div className="panel-title">
        <span>02 — CLASSIFICATION REPORT</span>

        <small className={result ? 'complete' : 'awaiting'}>
          {result
            ? '● COMPLETE'
            : status === 'running'
              ? '● ANALYZING'
              : '● AWAITING INPUT'}
        </small>
      </div>

      {status === 'running' ? (
        <div className="analysis-progress">
          <span>01 — Image preprocessing <b>✓</b></span>
          <span>02 — Feature extraction <b>✓</b></span>
          <span>03 — Model inference <i>→</i></span>
          <span>04 — Classification report</span>
        </div>
      ) : result ? (
        <div className="result-content">
          <div className="result-main">
            <p>PRIMARY CLASSIFICATION</p>

            <h3>
              {result.prediction.confidence < CONFIDENCE_THRESHOLD
                ? 'Other / Unrecognized Image'
                : result.prediction.class}
            </h3>

            {result.prediction.confidence < CONFIDENCE_THRESHOLD && (
              <p className="classification-warning">
                Confidence is below the acceptance threshold. This image does
                not meet the system's criteria for the six wound classes.
              </p>
            )}

            <div className="confidence-line">
              <span>CONFIDENCE SCORE</span>

              <b>
                {(result.prediction.confidence * 100).toFixed(1)}%
              </b>

              <i>
                <b
                  style={{
                    width: `${result.prediction.confidence * 100}%`,
                  }}
                />
              </i>
            </div>
          </div>

          <ConfidenceBars
            items={result.probabilities}
            predicted={
              result.prediction.confidence < CONFIDENCE_THRESHOLD
                ? ''
                : result.prediction.class
            }
          />

          <div className="model-meta">
            <span>
              <small>MODEL</small>
              {result.model.name}
            </span>

            <span>
              <small>OUTPUT CLASSES</small>
              {result.probabilities.length} classes
            </span>

            <span>
              <small>PROCESSING TIME</small>
              {result.processing_time_ms} ms
            </span>
          </div>
        </div>
      ) : (
        <div className="empty-result">
          <ScanLine />
          <p>Classification report will appear here after image analysis.</p>
          <small>MODEL READY · 6 OUTPUT CLASSES</small>
        </div>
      )}
    </section>
  )
}

export default ResultPanel
