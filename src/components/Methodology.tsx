const steps = [
  [
    '01',
    'Image input',
    'Source image is registered and prepared for controlled processing.',
  ],
  [
    '02',
    'Preprocessing',
    'Image is standardized for consistent analysis.',
  ],
  [
    '03',
    'Feature extraction',
    'Visual descriptors are translated into feature vectors.',
  ],
  [
    '04',
    'Model inference',
    'The classifier evaluates evidence against six categories.',
  ],
  [
    '05',
    'Confidence report',
    'A ranked, reviewable classification is generated.',
  ],
]

function Methodology() {
  return (
    <section className="method" id="method">
      <div>
        <p className="eyebrow"><i /> METHODOLOGY</p>

        <h2>
          Built around a
          <br />
          <em>disciplined</em> pipeline.
        </h2>
      </div>

      <div className="pipeline">
        {steps.map(([number, title, description]) => (
          <article key={number}>
            <span>{number}</span>

            <div>
              <h3>{title}</h3>
              <p>{description}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

export default Methodology
