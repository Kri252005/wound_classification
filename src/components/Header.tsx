import { ChevronRight, Menu, X } from 'lucide-react'
import { useState } from 'react'

function Header() {
  const [menuOpen, setMenuOpen] = useState(false)

  const closeMenu = () => {
    setMenuOpen(false)
  }

  return (
    <header>
      <a href="#top" className="logo" onClick={closeMenu}>
        <span className="mark"><span /></span>
        <span>Wound<span>Scope</span></span>
      </a>

      <nav className={menuOpen ? 'open' : ''}>
        <a href="#overview" onClick={closeMenu}>Overview</a>
        <a href="#classifier" onClick={closeMenu}>Classification</a>
        <a href="#method" onClick={closeMenu}>Methodology</a>
        <a href="#research" onClick={closeMenu}>Research</a>
        <a href="#about" onClick={closeMenu}>About</a>
      </nav>

      <div className="header-action">
        <a href="#classifier" className="open-link">
          Open classifier
          <ChevronRight size={14} />
        </a>

        <button
          className="menu"
          onClick={() => setMenuOpen((current) => !current)}
          aria-label="Toggle navigation"
          aria-expanded={menuOpen}
        >
          {menuOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>
    </header>
  )
}

export default Header
