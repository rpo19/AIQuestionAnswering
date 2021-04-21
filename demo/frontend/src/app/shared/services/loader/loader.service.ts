import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';

/**
 * Provides a service to subscribe to loading changes.
 */
@Injectable({
	providedIn: 'root'
})
export class LoaderService {

	private _loadingSubject: Subject<boolean> = new Subject();
	/**
	 * Observable which emits the loading status.
	 */
	readonly loading$ = this._loadingSubject.asObservable();

	constructor() { }

	/**
	 * Emits true to all subscribed observable.
	 */
	show(): void {
		this._loadingSubject.next(true);
	}

	/**
	 * Emits false to all subscribed observable.
	 */
	hide(): void {
		this._loadingSubject.next(false);
	}

}
