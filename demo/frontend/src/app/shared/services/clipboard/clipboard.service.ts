import { Inject, Injectable } from '@angular/core';
import { NAVIGATOR } from '@ng-web-apis/common';
import { from, Observable } from 'rxjs';


/**
 * Provides a service to copy text to clipboard
 */
@Injectable({
  providedIn: 'root'
})
export class ClipboardService {

  constructor(@Inject(NAVIGATOR) readonly navigator: Navigator) { }

  /**
   * Copy text to clipboard
   */
  copy(text: string): Observable<void> {
    return from(this.navigator.clipboard.writeText(text))
  }

}
