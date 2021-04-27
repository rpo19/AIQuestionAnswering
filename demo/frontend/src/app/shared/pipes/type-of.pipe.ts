import { NgModule, Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'typeOf'
})
export class TypeOfPipe implements PipeTransform {

  transform(value: any): any {
    return typeof value;
  }

}


@NgModule({
  imports: [],
  declarations: [TypeOfPipe],
  exports: [TypeOfPipe]
})
export class TypeOfPipeModule{}