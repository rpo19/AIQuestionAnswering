import { HttpEvent, HttpInterceptor, HttpHandler, HttpRequest, HttpResponse, HttpErrorResponse } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, throwError } from 'rxjs';
import { catchError, finalize } from 'rxjs/operators';
import { LoaderService } from '../services/loader/loader.service';

/**
 * Provides an interceptor to handle HTTP and client errors.
 */
@Injectable()
export class HttpLoaderInterceptor implements HttpInterceptor {

	constructor(
		private loaderService: LoaderService,
	) { }

	intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {

		this.loaderService.show();

		return next.handle(request).pipe(
			catchError((error: HttpErrorResponse) => {
				return throwError(error);
			}),
			finalize(() => this.loaderService.hide())
		);
	}
}
