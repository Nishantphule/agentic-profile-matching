# State Machine Diagram (Mermaid source)

> The repo ships the diagram in Mermaid form so it renders directly on
> GitHub / GitLab / Markdown viewers without needing a PNG.  To export to
> `state_machine.png`, paste the block below into <https://mermaid.live>
> and click **Download PNG**, or run:
>
> ```bash
> npx -y @mermaid-js/mermaid-cli -i diagrams/state_machine.md -o diagrams/state_machine.png
> ```

```mermaid
graph TD
    A([START]) --> B[Parse JD]
    B --> C[Extract Requirements]
    C --> D[Search Resumes - FAISS RAG]
    D -->|empty pool| Z([END])
    D --> E[Rank Candidates]
    E --> F[Generate Match Report]
    F --> G{Human Feedback (interrupt)}
    G -->|refine| H[Re-rank]
    H --> F
    G -->|approve| I[Final Recommendation]
    I --> Z([END])

    classDef interrupt fill:#FFD60A,stroke:#B58900,color:#222,stroke-width:2px;
    class G interrupt;
```
