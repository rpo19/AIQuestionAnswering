import { trigger, transition, style, animate } from '@angular/animations';
import { ChangeDetectionStrategy, Component, HostBinding, OnInit } from '@angular/core';
import {FormControl, FormGroup, Validators} from '@angular/forms';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { ApiService } from 'src/app/shared/api.service';
import { LoaderService } from 'src/app/shared/loader.service';
import { Data } from 'src/app/shared/models/data';

@Component({
  selector: 'app-asker',
  templateUrl: './asker.component.html',
  styleUrls: ['./asker.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AskerComponent implements OnInit {

  data$: Observable<Data>;
  loading$: Observable<boolean>;
  abstract$: Observable<string>;

  firstTime = true;
  mouseenter = false;

  searchForm = new FormGroup({
    question: new FormControl('')
  });

  get question(): FormControl {
    return this.searchForm.get('question') as FormControl;
  }

  constructor(
    private readonly _api: ApiService,
    private readonly _loader: LoaderService
  ) { }

  ngOnInit() {
    this.loading$ = this._loader.loading$;

    this._loader.loading$.subscribe(res => console.log(res));
  }

  ask(): void {
    const question = this.question.value;
    if (question) {
      this.data$ = this._api.ask_kgqa(this.searchForm.value.question)
      this.firstTime = false;
    }
  }

  getAbstract(entity: string): void {
    this.mouseenter = true;
    this.abstract$ = this._api.getAbstract(entity)
  }

}
