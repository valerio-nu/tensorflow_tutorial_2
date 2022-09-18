package TensorFlowGraph;

import org.tensorflow.*;

public class TensorClass {

    public static Graph createGraph() {
        Graph cgraph = new Graph();
        Operation a = cgraph.opBuilder("Const", "a").setAttr("dtype", DataType.fromClass(Double.class))
                .setAttr("value", Tensor.<Double>create(3.0, Double.class)).build();
        Operation b = cgraph.opBuilder("Const", "b").setAttr("dtype", DataType.fromClass(Double.class))
                .setAttr("value", Tensor.<Double>create(2.0, Double.class)).build();
        Operation x = cgraph.opBuilder("Placeholder", "x").setAttr("dtype", DataType.fromClass(Double.class)).build();
        Operation y = cgraph.opBuilder("Placeholder", "y").setAttr("dtype", DataType.fromClass(Double.class)).build();
        Operation ax = cgraph.opBuilder("Mul", "ax").addInput(a.output(0)).addInput(x.output(0)).build();
        Operation by = cgraph.opBuilder("Mul", "by").addInput(b.output(0)).addInput(y.output(0)).build();
        cgraph.opBuilder("Add", "z").addInput(ax.output(0)).addInput(by.output(0)).build();
        return cgraph;
    }

    public static Object runGraph(Graph graph, Double x, Double y) {
        Object result;
        try (Session newSession = new Session(graph)) {
            result = newSession.runner().fetch("z").feed("x", Tensor.<Double>create(x, Double.class))
                    .feed("y", Tensor.<Double>create(y, Double.class)).run().get(0).expect(Double.class)
                    .doubleValue();
        }
        return result;
    }

    public static void  main(String [] args){
        Graph graph = TensorClass.createGraph();
        Object result = TensorClass.runGraph(graph, 8.0, 6.0);
        System.out.println("The graph computation is as shown: " + result);
        graph.close();
    }
}
