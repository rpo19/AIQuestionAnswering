import { Inject, Injectable } from '@angular/core';
import { TOKENAPI } from '../tokens/token-api';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { Data } from './models/data';

@Injectable({
    providedIn: 'root'
})
export class ApiService {

	constructor(private _http: HttpClient, @Inject(TOKENAPI) private _api: string) { }

    ask_kgqa(question: string): Observable<Data> {
		return this._http.get<Data>(`${this._api}/api/kgqa?q=${question}`);
	}




	getAbstract(entity: string): Observable<any> {
		const query = `SELECT DISTINCT ?obj WHERE { ${entity} dbo:abstract ?obj . FILTER(LANG(?obj) = "en") } LIMIT 1`
	  
		return this._http.get<any>(`http://dbpedia.org/sparql?query=${query}`).pipe(map(res => res.results.bindings[0].obj.value.split('.')[0]));
	}

}
