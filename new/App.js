import React, { useState } from 'react'; // Correctly imported useState
import './App.css';
import CodeCard from './CodeCard';
import codeData from './data.js';
import logo from './logo.svg'; // Correctly import the logo

function App() {
    const [sortedData, setSortedData] = useState(codeData); // State initialized

    // Sorting function for "Most Used"
    const sortByUsage = () => {
        const sorted = [...sortedData].sort((a, b) => b.usage - a.usage);
        setSortedData(sorted);
    };

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
            <main>
                <div className="controls">
                    <button onClick={sortByUsage}>Sort by Most Used</button>
                </div>
                <section className="gallery">
                    {sortedData.map((code, index) => (
                        <CodeCard
                            key={index}
                            title={code.title}
                            description={code.description}
                            code={code.code}
                            dependencies={code.dependencies}
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
