import React, { useState } from 'react'; // Correctly imported useState
import './App.css';
import CodeCard from './CodeCard';
import codeData from './data.js';
import logo from './logo.svg'; // Correctly import the logo

function App() {
    const [sortedData, setSortedData] = useState(codeData); // State initialized
    const [searchTerm, setSearchTerm] = useState('');

    // Sorting function for "Most Used"
    const sortByUsage = () => {
        const sorted = [...sortedData].sort((a, b) => b.usage - a.usage);
        setSortedData(sorted);
    };
    const filteredData = sortedData.filter(code =>
    code.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    code.description.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className="App">
            <header className="App-header">
                <a href="/" className="logo">
                    <img src={logo} alt="Spark leArnZ Logo" title="Home"/>
                </a>
                <a href="/" className="home-link">
                    <h1>Spark learnZ</h1>
                </a>
            </header>
            
            <div className="search-container">
                <button onClick={() => setSearchTerm('')} className="reset-btn">Reset</button>
                <input 
                    type="text" 
                    id="searchBox" 
                    placeholder="Search topics..." 
                    value={searchTerm} 
                    onChange={(e) => setSearchTerm(e.target.value)} 
                />
            </div>

            <main>
                <div className="controls">
                    <button onClick={sortByUsage}>Sort by Most Used</button>
                </div>
                <section className="gallery">
                    {filteredData.map((code, index) => (
                        <CodeCard
                            key={index}
                            title={code.title}
                            description={code.description}
                            code={code.code}
                            dependencies={code.dependencies}
                            searchTerm={searchTerm}
                        />
                    ))}
                </section>
            </main>
            <footer>
                <p>Welcome to the PySpark learning platform!</p>
            </footer>
        </div>
    );
}
export default App;
 