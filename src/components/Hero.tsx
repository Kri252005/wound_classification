import { ArrowUpRight, ChevronRight, FileScan, FlaskConical, ScanLine } from 'lucide-react'

function Button({
  children,
  secondary = false,
}: {
  children: React.ReactNode
  secondary?: boolean
}) {
  return (
    <span className={secondary ? 'button secondary' : 'button'}>
      {children}
      <ArrowUpRight size={15} />
    </span>
  )
}

function EvidenceImage() {
  return (
    <div className="evidence-image">
      <div className="blood-mark one" />
      <div className="blood-mark two" />
      <div className="blood-mark three" />
      <div className="scan-hairline" />

      <span className="corner tl" />
      <span className="corner tr" />
      <span className="corner bl" />
      <span className="corner br" />

      <span className="coord x">X: 440.64</span>
      <span className="coord y">Y: 187.28</span>
      <span className="ruler top-ruler">0&nbsp;&nbsp; 10&nbsp;&nbsp; 20&nbsp;&nbsp; 30&nbsp;&nbsp; 40</span>
    </div>
  )
}

function Hero() {
  return (
    <>
      <section className="hero" id="overview">
        <div className="hero-copy">
          <p className="eyebrow"><i /> FORENSIC ML SYSTEM · v1.0</p>

          <h1>
            Evidence, <em>examined.</em>
            <br />
            Wounds, classified.
          </h1>

          <p className="lede">
            An image-based classification system designed to assist the
            analysis of wound patterns across clinically relevant categories.
          </p>

          <div className="actions">
            <a href="#classifier">
              <Button>Start classification</Button>
            </a>

            <a href="#method">
              <Button secondary>Explore methodology</Button>
            </a>
          </div>

          <p className="session">
            SESSION / WS-24-0917 · SECURE RESEARCH ENVIRONMENT
          </p>
        </div>

        <div className="hero-visual">
          <div className="visual-heading">
            <span>LIVE ANALYSIS VIEW</span>
            <span>01 / 01</span>
          </div>

          <EvidenceImage />

          <div className="visual-footer">
            <div>
              <small>PRIMARY FINDING</small>
              <strong>Laceration</strong>
            </div>

            <div className="confidence">
              <small>CONFIDENCE</small>
              <strong>92.4%</strong>
              <i><b /></i>
            </div>

            <ScanLine className="scan-icon" />
          </div>

          <span className="serial">EVIDENCE IMAGE / IMG-2039-A</span>
        </div>
      </section>

      <section className="system-strip">
        <div>
          <b>06</b>
          <span>
            <small>CLASSIFICATION CATEGORIES</small>
            Abrasions · Bruises · Burns · Cuts · Lacerations · Stab Wounds
          </span>
        </div>

        <div>
          <FileScan />
          <span>
            <small>MACHINE LEARNING PIPELINE</small>
            Image → Feature extraction → Classification → Prediction
          </span>
        </div>

        <div>
          <FlaskConical />
          <span>
            <small>RESEARCH-ORIENTED</small>
            Designed for experimental analysis and academic research.
          </span>
        </div>
      </section>
    </>
  )
}

export default Hero
