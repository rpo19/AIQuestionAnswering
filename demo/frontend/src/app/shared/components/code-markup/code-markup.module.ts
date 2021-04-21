import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CodeMarkupComponent } from './code-markup.component';
import { TuiButtonModule } from '@taiga-ui/core';

@NgModule({
  imports: [
    CommonModule,
    TuiButtonModule
  ],
  declarations: [CodeMarkupComponent],
  exports: [CodeMarkupComponent]
})
export class CodeMarkupModule { }
