import Header from './components/Header'
import Hero from './components/Hero'
import Classifier from './components/Classifier'
import Methodology from './components/Methodology'
import Research from './components/Research'

function App() {
  return (
    <main id="top">
      <Header />
      <Hero />
      <Classifier />
      <Methodology />
      <Research />

      <footer id="about">
        <a href="#top" className="logo footer-logo">
          <span className="mark"><span /></span>
          <span>Wound<span>Scope</span></span>
        </a>

        <div>
          <b>Wound Classification &amp; Forensic Analysis</b>
          <p>
            For research and educational purposes only. This system does not
            provide medical diagnosis.
          </p>
        </div>

        <small>
          © 2026 WOUNDSCOPE<br />
          MACHINE LEARNING RESEARCH PROJECT
        </small>
      </footer>
    </main>
  )
}

export default App
