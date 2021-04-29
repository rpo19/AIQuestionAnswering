import { ChangeDetectionStrategy, Component, Inject, OnInit } from '@angular/core';
import { FormControl, FormGroup } from '@angular/forms';
import { TuiDialogContext, TuiDialogService, TuiNotification, TuiNotificationsService } from '@taiga-ui/core';
import { PolymorpheusTemplate } from '@tinkoff/ng-polymorpheus';
import { Observable, of } from 'rxjs';
import { ApiService } from 'src/app/shared/services/api/api.service';
import { LoaderService } from 'src/app/shared/services/loader/loader.service';
import { DataFTQA, DataKGQA } from 'src/app/shared/models/data';
import { catchError, debounceTime, distinctUntilChanged, filter, map, startWith, switchMapTo, takeUntil, tap } from 'rxjs/operators';
import { QuestionStoreService } from 'src/app/shared/services/local-storage/question-store.service';
import { HttpErrorResponse } from '@angular/common/http';
import { TuiDestroyService } from '@taiga-ui/cdk';
import { floor } from '@taiga-ui/cdk';

@Component({
  selector: 'app-asker',
  templateUrl: './asker.component.html',
  styleUrls: ['./asker.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
  providers: [TuiDestroyService]
})
export class AskerComponent implements OnInit {

  // emits responses from kgqa
  dataKGQA$: Observable<DataKGQA | void>;
  // emits responses from ftqa
  dataFTQA$: Observable<DataFTQA | void>;
  // emit loading status on request
  loading$: Observable<boolean>;
  // emit previous questions stored in the local storage
  previousQuestions$: Observable<string[]>;

  firstTime = true;

  // keeps track of dropdown status
  openDropdown = false;

  optionsFTQA = [
    'NER and NEL',
    'Span of text',
    'Wikipedia search'
  ];


  // search form
  searchForm = new FormGroup({
    question: new FormControl(''),
    type: new FormControl('kgqa'),
    ftqaType: new FormControl(this.optionsFTQA[0]),
    span: new FormControl('')
  });



  constructor(
    private readonly _api: ApiService,
    private readonly _loader: LoaderService,
    private readonly _questionStoreService: QuestionStoreService,
    @Inject(TuiDestroyService) private readonly destroy$: TuiDestroyService,
    @Inject(TuiNotificationsService) private readonly _notificationsService: TuiNotificationsService,
    @Inject(TuiDialogService) private readonly dialogService: TuiDialogService
  ) { }

  ngOnInit() {
    this.loading$ = this._loader.loading$;

    this.previousQuestions$ = this.question.valueChanges.pipe(
      debounceTime(200),
      distinctUntilChanged(),
      switchMapTo(this._questionStoreService.get('questions')),
      filter(questions => questions),
      map((questions: string[]) => {
        const foundQ = this._filterQuestions(questions);
        return !foundQ || foundQ.length === 0 ? [] : foundQ
      }),
      startWith([]),
      tap((questions) => this._shouldCloseDropdown(questions) ? this.openDropdown = false : this.openDropdown = true),
      takeUntil(this.destroy$)
    )
  }

  private _filterQuestions(questions: string[]): string[] {
    const val = this.question.value.toLowerCase()
    return questions.filter(question => question.toLowerCase().includes(val)).slice(0, 5);
  }

  private _shouldCloseDropdown(questions: string[]): boolean {
    return questions.length === 0 || questions.length === 1 &&
      questions[0].toLowerCase() === this.question.value.toLowerCase();
  }

  /**
  * Getter question form control
  */
  get question(): FormControl {
    return this.searchForm.get('question') as FormControl;
  }

  /**
  * Getter type form control
  */
  get type(): FormControl {
    return this.searchForm.get('type') as FormControl;
  }

  /**
  * Getter type form control
  */
  get ftqaType(): FormControl {
    return this.searchForm.get('ftqaType') as FormControl;
  }

  /**
   * Handle ask
   */
  handleAsk(question: string): void {
    if (question) {
      if (this.type.value === 'kgqa') {
        this.dataFTQA$ = of(null);
        this.dataKGQA$ = this._api.ask_kgqa(this.searchForm.value.question).pipe(
          catchError((err: HttpErrorResponse) => {
            return this._notificationsService.show(err.error.error, { status: TuiNotification.Error })
          }),
          tap(() => this._questionStoreService.set('questions', this.question.value))
        );
      } else {
        // retrieve answer for ftqa
        this.dataKGQA$ = of(null);

        this.dataFTQA$ = this._api.ask_ftqa(this.searchForm.value).pipe(
          catchError((err: HttpErrorResponse) => {
            return this._notificationsService.show(err.error.error, { status: TuiNotification.Error })
          }),
          tap(() => this._questionStoreService.set('questions', this.question.value))
        );
      }
      this.firstTime = false;
    }
  }

  /**
   * Ask on form submit
   */
  ask(): void {
    const question = this.question.value;
    this.handleAsk(question);
  }

  /**
   * Opens SPARQL query dialog
   */
  openDialog(content: PolymorpheusTemplate<TuiDialogContext>): void {
    this.dialogService.open(content, { size: 'l' }).subscribe();
  }

  /**
   * Set question selected from dropdown
   */
  setQuestion(prevQuestion: string) {
    this.searchForm.patchValue({
      question: prevQuestion
    })
    this.openDropdown = false;
  }

  /**
   * Get a link from a resource
   */
  getLink(resource: string): string {
    return resource.split('<')[1].split('>')[0];
  }

  round(value: number): number {
    return floor(value, 3);
  }

  transformSectionLink(section: string): string {
    return section.replaceAll(' ', '_',);
  }

}
