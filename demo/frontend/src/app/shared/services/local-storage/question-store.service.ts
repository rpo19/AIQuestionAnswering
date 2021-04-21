import { Injectable } from '@angular/core';
import { StorageMap } from '@ngx-pwa/local-storage';
import { Observable } from 'rxjs';
import { filter, switchMap, tap } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class QuestionStoreService {

  constructor(private storage: StorageMap) { }

  /**
   * Add a new question to local storage
   */
  set(key: string, newQuestion: string): void {
    this.storage.get(key).pipe(
      filter((data: string[]) => data && !data.some(el => el === newQuestion) || data == null),
      switchMap((data: string[]) => (
        this.storage.set(key, data ? [newQuestion, ...data] : [newQuestion]))
      )
    ).subscribe();
  }

  /**
   * Get questions array from local storage
   */
  get(key: string): Observable<any> {
    return this.storage.get(key);
  }

  watchQuestions(key: string): Observable<any> {
    return this.storage.watch(key) as Observable<string[]>
  }

}
