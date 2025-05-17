import React, { useState } from 'react';
import './CodeCard.css';

function CodeCard({ title, description, code, dependencies = [], searchTerm }) {
    const [expanded, setExpanded] = useState(false);

    return (
        <div className={`card ${expanded ? "expanded" : ""}`} onClick={() => setExpanded(!expanded)}>
            <h3 dangerouslySetInnerHTML={{ __html: highlightText(title, searchTerm) }}></h3>
            <p dangerouslySetInnerHTML={{ __html: highlightText(description, searchTerm) }}></p>


            {expanded && (
                <>
                    <div className="expanded-content">
                        <pre className="slide-up">
                            <code>{code}</code>
                        </pre>

                        {/* Display Dependencies below description */}
                        {dependencies.length > 0 && (
                            <div className="dependencies">
                                <h4>Dependencies:</h4>
                                <ul>
                                    {dependencies.map((dep, index) => (
                                        <li key={index}>{dep}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        <button className="close-btn" onClick={(e) => { e.stopPropagation(); setExpanded(false); }}>
                            ✖ Close
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}

function highlightText(text, searchTerm) {
    if (!searchTerm) return text;
    
    const regex = new RegExp(`(${searchTerm})`, 'gi'); // Case-insensitive match
    return text.replace(regex, '<span class="highlight">$1</span>');
}

export default CodeCard;
 