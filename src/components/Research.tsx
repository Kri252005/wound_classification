const metrics = [
  ['92.4%', 'Accuracy'],
  ['0.94', 'ROC-AUC'],
  ['91.8%', 'Precision'],
  ['90.7%', 'Recall'],
]

function Research() {
  return (
    <section className="research" id="research">
      <div className="section-head">
        <div>
          <p className="eyebrow"><i /> RESEARCH PERFORMANCE</p>

          <h2>
            Measured, not <em>marketed.</em>
          </h2>
        </div>

        <span className="analysis-id">
          VALIDATION SET · N=1,280
        </span>
      </div>

      <div className="metrics">
        {metrics.map(([value, label]) => (
          <div key={label}>
            <strong>{value}</strong>
            <small>{label}</small>
          </div>
        ))}
      </div>
    </section>
  )
}

export default Research
