import {
  ChangeDetectionStrategy,
  Component,
  Input,
  OnInit
} from '@angular/core';
import { merge, Observable, of, Subject, timer } from 'rxjs';
import { mapTo, startWith, switchMap, switchMapTo } from 'rxjs/operators';
import { ClipboardService } from '../../services/clipboard/clipboard.service';

/**
 * Define a row of code text
 */
interface Row {
  words: string[];
}

/**
 * Define operators SPARQL classes
 */
const OPERATORS = {
  SELECT: 'word-select',
  ASK: 'word-ask',
  COUNT: 'word-count',
  DISTINCT: 'word-distinct',
  WHERE: 'word-where'
};

/**
 * A component do display a SPARQL query
 */
@Component({
  selector: 'app-code-markup',
  templateUrl: './code-markup.component.html',
  styleUrls: ['./code-markup.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class CodeMarkupComponent implements OnInit {
  // content 
  @Input() content: string;

  // Subject to next new copy events
  private _copy$$: Subject<void> = new Subject();

  // Emits copy status
  readonly copy$: Observable<boolean> = this._copy$$.pipe(
    switchMap(() => this._clipboardService.copy(this.content)),
    switchMapTo(
      merge(
        of(false),
        timer(1000).pipe(mapTo(true))
      )),
    startWith(true)
  );
  
  // rows of text
  rows: Row[];
  // operators to style
  operators = OPERATORS;

  constructor(
    private readonly _clipboardService: ClipboardService
  ) {}

  ngOnInit() {
    this.rows = this._parseSPARQL(this.content); 
  }

  /**
   * Parse SPARQL query to format text in rows of words
   */
  private _parseSPARQL(query: string): Row[] {
    const rowsString = query.split('\n');
    const rows: Row[] = rowsString.map(rowString => ({
      words: rowString.split(' ')
    }));
    return rows;
  }

  /**
   * Trigger events of copy on click
   */
  copyToClipboard(): void {
    this._copy$$.next();
  }
}
