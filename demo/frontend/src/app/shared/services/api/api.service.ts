import { Inject, Injectable } from '@angular/core';
import { TOKENAPI } from '../../../tokens/token-api';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { DataFTQA, DataKGQA } from '../../models/data';

@Injectable({
    providedIn: 'root'
})
export class ApiService {

	constructor(private _http: HttpClient, @Inject(TOKENAPI) private _api: string) { }

    ask_kgqa(question: string): Observable<DataKGQA> {
		return this._http.get<DataKGQA>(`${this._api}api/kgqa?q=${question}`);
	}

	ask_ftqa(formValues: any): Observable<DataFTQA> {
		let options = `&mode=${formValues.ftqaType}`
		if (formValues.ftqaType === 'Span of text') {
			options += `&span=${formValues.span}`
		}
		console.log(options)
		return this._http.get<DataFTQA>(`${this._api}api/ftqa?q=${formValues.question + options}`);
	}

}
