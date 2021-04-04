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

}
