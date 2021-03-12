package com.mycompany.app;

import org.apache.jena.query.QueryFactory;
import org.apache.jena.sparql.core.TriplePath;
import org.apache.jena.sparql.syntax.ElementPathBlock;
import org.apache.jena.sparql.syntax.ElementVisitorBase;
import org.apache.jena.sparql.syntax.ElementWalker;
import org.json.JSONArray;
import org.json.JSONObject;
import org.apache.jena.query.Query;
import org.apache.jena.graph.Node;
import org.apache.jena.query.QueryParseException;

import java.util.Set;
import java.util.HashSet;
import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.LinkedList;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.jena.graph.NodeFactory;

import picocli.CommandLine;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.util.concurrent.Callable;

@Command(
  name = "Sparql2Graph",
  version = "Sparql v1.0",
  mixinStandardHelpOptions = true, // add --help and --version options
  description = "Extracts Structural query graph from SPARQL queries.\n"+
    "It reads queries from '--input-file' and then prints query graph to stdout.\n"+
    "Usage:\n"+
    "java -cp file.jar com.mycompany.app.App --input-file /path/to/input-file.csv > /path/to/output-file.csv"
)
class Main implements Callable<Integer> {

    @Option(
        names = {"-i", "--input-file"},
        paramLabel = "FILE",
        description = "Input file containing SPARQL queries one per line."
    )
    private String inputFile;

    @Override
    public Integer call() {

        if (inputFile == null) {
            System.err.println("ERROR: '--input-file' must be specified.");
            return 1;
        }

        Pattern pattern = Pattern.compile("^[\"\\s]+(.*)[\"\\s]+$");
        Matcher matcher;
        boolean matchFound;

        final Node type = NodeFactory.createURI("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");

        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader(inputFile));
            String line = reader.readLine();
            int lineCount = 0;
            while (line != null) {
                lineCount++;
                // System.out.println(line);
                Query q;
                matcher = pattern.matcher(line);
                matchFound = matcher.find();
                if (matchFound) {
                    line = matcher.group(1);
                }

                try {
                    q = QueryFactory.create(line);
                } catch (QueryParseException e) {
                    System.err.println("Error on line " + lineCount);
                    System.err.println(line);
                    // e.printStackTrace();
                    // read next line
                    line = reader.readLine();
                    continue;
                }
                // System.out.println(q);

                final Set<TriplePath> triplesSet = new HashSet<TriplePath>();
                final Map<Node, List<Node>> graphMap = new HashMap<Node, List<Node>>();

                // https://stackoverflow.com/questions/15203838/how-to-get-all-of-the-subjects-of-a-jena-query
                // This will walk through all parts of the query
                ElementWalker.walk(q.getQueryPattern(),
                        // For each element...
                        new ElementVisitorBase() {
                            // ...when it's a block of triples...
                            public void visit(ElementPathBlock el) {
                                // ...go through all the triples...
                                Iterator<TriplePath> triples = el.patternElts();
                                while (triples.hasNext()) {
                                    TriplePath triple = triples.next();
                                    triplesSet.add(triple);
                                }
                            }
                        });

                for (TriplePath triple : triplesSet) {
                    if (!triple.getPredicate().equals(type)) {

                        List<Node> currentSet = graphMap.get(triple.getSubject());
                        if (currentSet == null) {
                            currentSet = new LinkedList<Node>();
                            currentSet.add(triple.getObject());
                            graphMap.put(triple.getSubject(), currentSet);
                        } else {
                            currentSet.add(triple.getObject());
                        }
                    } else {
                        // System.err.println("Found type!!");
                        // System.err.println(triple.getPredicate());

                        List<Node> currentSet = graphMap.get(triple.getSubject());
                        if (currentSet == null) {
                            currentSet = new LinkedList<Node>();
                            graphMap.put(triple.getSubject(), currentSet);
                        }
                    }
                }

                JSONObject outJSON = new JSONObject();

                // System.out.println(new JSONObject(graphMap));
                for (Map.Entry<Node, List<Node>> entry : graphMap.entrySet()) {
                    JSONArray currentArray = new JSONArray();
                    for (Node n : entry.getValue()) {
                        currentArray.put(n.toString());
                    }
                    outJSON.put(entry.getKey().toString(), currentArray);
                    // ...
                }

                System.out.println(outJSON.toString());

                // read next line
                line = reader.readLine();
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return 0;
    }

}

public class App {
    public static void main(String[] args) {
        System.exit(new CommandLine(new Main()).execute(args));
    }
}
