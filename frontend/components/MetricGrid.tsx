type MetricGridProps = {
  metrics: Record<string, number>;
  confidence: number | null;
};

export function MetricGrid({ metrics, confidence }: MetricGridProps) {
  const entries = Object.entries(metrics);
  return (
    <section className="metrics" aria-label="Metrics">
      {confidence !== null && (
        <div className="metric">
          <span>Confidence</span>
          <strong>{Math.round(confidence * 100)}%</strong>
        </div>
      )}
      {entries.map(([name, value]) => (
        <div className="metric" key={name}>
          <span>{name.toUpperCase()}</span>
          <strong>{Number.isFinite(value) ? value.toFixed(4) : "0.0000"}</strong>
        </div>
      ))}
    </section>
  );
}
