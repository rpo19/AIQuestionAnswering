/**
 * Response from querying kgqa endpoint
 */
export interface DataKGQA {
    pattern: string;
    entities: string;
    answers: string[];
    graph: any;
    query: string;
}