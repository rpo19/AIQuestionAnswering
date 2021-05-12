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

    @Option(
        names = {"-f", "--format"},
        paramLabel = "FORMAT",
        description = "Output format of the graph; between \"dict_of_lists\" (default)" +
                        "and \"dict_of_dicts\" (used in evaluation)."
    )
    private String format;

    private int FORMAT_DICT_OF_LISTS = 0;
    private int FORMAT_DICT_OF_DICTS = 1;
    private int FORMAT = FORMAT_DICT_OF_LISTS;

    @Override
    public Integer call() {

        if (inputFile == null) {
            System.err.println("ERROR: '--input-file' must be specified.");
            return 1;
        }

        if (format != null && format.equals("dict_of_dicts")) {
            FORMAT = FORMAT_DICT_OF_DICTS;
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
                // dict of dicts
                final HashMap<Node, HashMap<Node, String>> graphMapDoD = new HashMap<Node, HashMap<Node, String>>();

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

                        // dict of lists
                        List<Node> currentSet = graphMap.get(triple.getSubject());
                        if (currentSet == null) {
                            currentSet = new LinkedList<Node>();
                            currentSet.add(triple.getObject());
                            graphMap.put(triple.getSubject(), currentSet);
                        } else {
                            currentSet.add(triple.getObject());
                        }

                        // dict of dicts
                        // node -> outerMap(node, predicate)
                        HashMap<Node, String> outerMap = graphMapDoD.get(triple.getSubject());
                        if (outerMap == null) {
                            outerMap = new HashMap<Node, String>();
                            graphMapDoD.put(triple.getSubject(), outerMap);
                        }
                        // predicate label
                        String res = outerMap.put(triple.getObject(), triple.getPredicate().toString());
                        if (res != null) {
                            System.err.println("WARNING: label was already present in predicateMap for " + triple.toString());
                        }
                    } else {
                        // System.err.println("Found type!!");
                        // System.err.println(triple.getPredicate());

                        // dict of lists
                        List<Node> currentSet = graphMap.get(triple.getSubject());
                        if (currentSet == null) {
                            currentSet = new LinkedList<Node>();
                            graphMap.put(triple.getSubject(), currentSet);
                        }

                        // dict of dicts
                        HashMap<Node, String> outerMap = graphMapDoD.get(triple.getSubject());
                        if (outerMap == null) {
                            outerMap = new HashMap<Node, String>();
                            graphMapDoD.put(triple.getSubject(), outerMap);
                        }
                    }
                }

                JSONObject outJSON = new JSONObject();

                // System.out.println(new JSONObject(graphMap));
                if (FORMAT == FORMAT_DICT_OF_LISTS) {
                    for (Map.Entry<Node, List<Node>> entry : graphMap.entrySet()) {
                        JSONArray currentArray = new JSONArray();
                        for (Node n : entry.getValue()) {
                            currentArray.put(n.toString());
                        }
                        outJSON.put(entry.getKey().toString(), currentArray);
                        // ...
                    }
                }
                else if (FORMAT == FORMAT_DICT_OF_DICTS) {
                    for (Map.Entry<Node, HashMap<Node, String>> level1 : graphMapDoD.entrySet()) {
                        // for each subject
                        JSONObject lvl1JSON = new JSONObject();
                        for (Map.Entry<Node, String> level2 : level1.getValue().entrySet()) {
                            // for each object
                            // object representing predicate: contains label
                            JSONObject lvl2JSON = new JSONObject();
                            lvl2JSON.put("label", level2.getValue());

                            lvl1JSON.put(level2.getKey().toString(), lvl2JSON);
                        }
                        outJSON.put(level1.getKey().toString(), lvl1JSON);
                    }
                }
                else {
                    System.err.println("ERROR: invalid format.");
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
