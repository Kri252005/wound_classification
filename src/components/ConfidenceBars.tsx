import type { Probability } from '../types/prediction'

type ConfidenceBarsProps = {
  items: Probability[]
  predicted: string
}

function ConfidenceBars({
  items,
  predicted,
}: ConfidenceBarsProps) {
  const sortedItems = [...items].sort(
    (a, b) => b.probability - a.probability
  )

  return (
    <div className="bars">
      {sortedItems.map((item) => (
        <div
          className={`bar-row ${item.class === predicted ? 'active' : ''}`}
          key={item.class_index}
        >
          <span>{item.class}</span>

          <div className="bar-track">
            <i
              style={{
                width: `${item.probability * 100}%`,
              }}
            />
          </div>

          <b>{(item.probability * 100).toFixed(1)}%</b>
        </div>
      ))}
    </div>
  )
}

export default ConfidenceBars
