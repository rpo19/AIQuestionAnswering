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

export interface DataFTQA {
    answers: AnswerFTQA[];
    answerType: string;
}

export interface AnswerFTQA {
    answer: string;
    section: string;
    start: number;
    end: number;
    score: number;
    entity: string;
}