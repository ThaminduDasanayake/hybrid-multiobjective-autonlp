import { Card, CardContent } from "@/components/ui/card.jsx";

const MetricCard = ({ icon: Icon, label, value, sub }) => {
  return (
    <Card className="bg-card shadow-sm border-border">
      <CardContent className="p-5">
        <div className="flex flex-row items-center justify-between pb-2">
          <h3 className="text-sm font-medium text-muted-foreground">{label}</h3>
          <Icon size={18} className="text-muted-foreground" />
        </div>
        <div>
          <p className="text-3xl font-bold tracking-tight text-foreground">{value}</p>
          {sub && <p className="mt-1 text-xs text-muted-foreground">{sub}</p>}
        </div>
      </CardContent>
    </Card>
  );
};
export default MetricCard;
